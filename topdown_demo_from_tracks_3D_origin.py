#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topdown_demo_from_tracks_3D.py
- 从 tracks_dir/track_*.csv 读取人框与 tid，跑 **一次** 2D Top-Down，并写出 2D 可视化视频（保持原行为）
- 若提供 --pose-lifter-config 与 --pose-lifter-checkpoint，则将这一次的 2D 结果
  直接喂入官方 3D Pose Lifter 流程（时序抽窗 → lifter），默认不改变画面，可选 --save-3d-json 落盘

使用示例：
python topdown_demo_from_tracks_3D.py ^
  --pose-config ./configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py ^
  --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth ^
  --pose-lifter-config ./configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py ^
  --pose-lifter-checkpoint ./checkpoints/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.pth ^
  --video ./video/mabei.mp4 ^
  --tracks-dir ./mabei_yolo_tranreid_multibody_kalman ^
  --out ./video/mabei_pose_from_tracks_3d.mp4 ^
  --save-3d-json ./video/out_3d.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from mmengine.structures import InstanceData
import cv2
import mmcv
import mmengine
import numpy as np
from tqdm.auto import tqdm

# ===== 2D（官方 1.x API）=====
from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis import inference_topdown
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, PoseDataSample

# ===== 3D（严格采用官方 body3d_pose_lifter_demo 的调用方式）=====
from mmpose.apis import (
    init_model as init_model_official,  # lifter 也用 init_model
    convert_keypoint_definition,
    extract_pose_sequence,
    inference_pose_lifter_model,
)

# ----------------- 工具函数 -----------------
def _expand_bbox(b: List[float], H: int, W: int, scale: float = 1.15, extra: int = 4) -> List[int]:
    """适度放大 bbox，避免裁剪肢体；返回 clamp 后的整数 xyxy。"""
    x1, y1, x2, y2 = map(float, b)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    nw, nh = w * scale + 2 * extra, h * scale + 2 * extra
    nx1 = max(0, int(round(cx - nw / 2.0)))
    ny1 = max(0, int(round(cy - nh / 2.0)))
    nx2 = min(W, int(round(cx + nw / 2.0)))
    ny2 = min(H, int(round(cy + nh / 2.0)))
    if nx2 <= nx1: nx2 = min(W, nx1 + 1)
    if ny2 <= ny1: ny2 = min(H, ny1 + 1)
    return [nx1, ny1, nx2, ny2]


def load_tracks(tracks_dir: str) -> Tuple[Dict[int, List[Dict[str, Any]]], int]:
    """读取 tracks_dir 下的 track_*.csv，聚合为 frame_index -> [{bbox, score, tid}, ...]"""
    csv_paths = sorted(glob.glob(os.path.join(tracks_dir, "track_*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"未在 {tracks_dir} 下找到 track_*.csv")

    frame_map: Dict[int, List[Dict[str, Any]]] = {}
    max_frame = 0
    for p in csv_paths:
        base = os.path.basename(p)
        try:
            stem = os.path.splitext(base)[0]
            tid = int(stem.split("_")[-1])
        except Exception:
            tid = 0

        with open(p, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fi = int(float(row["frame"]))
                except Exception:
                    continue

                x1 = row["x1"]; y1 = row["y1"]; x2 = row["x2"]; y2 = row["y2"]
                if x1 in (None, "", "None") or y1 in (None, "", "None") or x2 in (None, "", "None") or y2 in (None, "", "None"):
                    max_frame = max(max_frame, fi)
                    continue

                try:
                    b = [float(x1), float(y1), float(x2), float(y2)]
                except Exception:
                    continue

                sc = row.get("score", "")
                try:
                    score = float(sc) if sc not in ("", "None", None) else 1.0
                except Exception:
                    score = 1.0

                item = {"bbox": b, "score": score, "tid": tid}
                frame_map.setdefault(fi, []).append(item)
                max_frame = max(max_frame, fi)
    return frame_map, max_frame


# ----------------- 主流程 -----------------
def run_video_with_tracks(
    pose_config: str,
    pose_checkpoint: str,
    video_path: str,
    tracks_dir: str,
    out_path: Optional[str] = None,
    device: str = "cuda:0",
    kpt_score_thr: float = 0.3,
    expand: float = 1.15,
    extra: int = 4,
    show: bool = False,
    radius: int = 4,
    thickness: int = 2,
    alpha: float = 0.8,
    draw_bbox: bool = True,
    show_progress: bool = True,
    # ===== 3D：全部可选；不提供则不启用 =====
    pose_lifter_config: Optional[str] = None,
    pose_lifter_checkpoint: Optional[str] = None,
    seq_len: Optional[int] = None,    # 若不显式给，按 lifter 的数据集默认
    save_3d_json: Optional[str] = None,
    disable_rebase_keypoint: bool = False,
    disable_norm_pose_2d: bool = False,
):
    # 1) 初始化 2D 姿态模型
    pose_estimator = init_pose_estimator(
        pose_config,
        pose_checkpoint,
        device=device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
    )
    # 2D 可视化器
    pose_estimator.cfg.visualizer.radius = radius
    pose_estimator.cfg.visualizer.alpha = alpha
    pose_estimator.cfg.visualizer.line_width = thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style="mmpose")

    # 2) 若提供 3D 模型，则初始化 lifter 与配置
    use_3d = bool(pose_lifter_config and pose_lifter_checkpoint)
    if use_3d:
        pose_lifter = init_model_official(
            pose_lifter_config, pose_lifter_checkpoint, device=device
        )
        pose_est_results_list: List[List[PoseDataSample]] = []  # 缓存每帧“已转换”的 2D 结果
        results3d: Dict[int, Dict[int, Dict[str, Any]]] = {}

        pose_det_dataset_name = pose_estimator.dataset_meta.get('dataset_name', None)
        pose_lift_dataset_name = pose_lifter.dataset_meta.get('dataset_name', None)

        ds_cfg = pose_lifter.cfg.test_dataloader.dataset
        if seq_len is None:
            seq_len = ds_cfg.get('seq_len', 1)
        seq_step = ds_cfg.get('seq_step', 1)
        causal = ds_cfg.get('causal', False)
        print(f"[3D lifter] dataset={pose_lifter.dataset_meta.get('dataset_name')}, "
        f"seq_len={seq_len}, step={seq_step}, causal={causal}")
    
    # 3) 读取 CSV、人框驱动（不变）
    frame_map, _ = load_tracks(tracks_dir)

    # 4) 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=(total_frames if total_frames > 0 else None),
                unit="frame", disable=not show_progress)

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # === A) 2D 推理 + 可视化（保持原逻辑） ===
        items = frame_map.get(frame_idx, [])
        bxs, tids = [], []
        for it in items:
            bx = _expand_bbox(it["bbox"], H, W, scale=expand, extra=extra)
            bxs.append(bx)
            tids.append(int(it.get("tid", 0)))

        if len(bxs) == 0:
            vis_img = frame.copy()
            pose_results: List[PoseDataSample] = []
        else:
            bboxes = np.asarray(bxs, dtype=np.float32)  # [N,4]
            pose_results = inference_topdown(pose_estimator, frame, bboxes)
            for ds in pose_results:
                if not hasattr(ds, 'gt_instances'):
                    ds.set_field(InstanceData(), 'gt_instances')
            data_samples = merge_data_samples(pose_results)

            rgb = mmcv.bgr2rgb(frame)
            visualizer.add_datasample(
                "result", rgb, data_sample=data_samples,
                draw_gt=False, draw_heatmap=False, draw_bbox=draw_bbox,
                show=False, wait_time=0, kpt_thr=kpt_score_thr,
            )
            vis_img = mmcv.rgb2bgr(visualizer.get_image())

            # 叠加 CSV 的 track id
            for bx, tid in zip(bxs, tids):
                x1, y1, x2, y2 = map(int, bx)
                cv2.putText(vis_img, f"ID{tid}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # === B) 3D：直接复用本帧 2D 结果进入官方 lifter 流程（无二次 2D/无二次跟踪） ===
        if use_3d and len(pose_results) > 0:
            # 1) 将当帧 2D 结果转换为 3D 所需骨架定义，并带上 track_id = CSV 的 tid
            converted_list: List[PoseDataSample] = []
            for tid, bx, ds in zip(tids, bxs, pose_results):
                pred_np = ds.pred_instances.cpu().numpy()
                keypoints = pred_np.keypoints  # [K,3]

                # 若 pred_instances 中没有 bboxes/areas，则用 CSV bbox 补齐
                if not hasattr(ds.pred_instances, 'bboxes'):
                    ds.pred_instances.set_field(np.asarray([bx], dtype=np.float32), 'bboxes')
                if not hasattr(ds.pred_instances, 'areas'):
                    x1, y1, x2, y2 = bx
                    ds.pred_instances.set_field(
                        np.asarray([(x2 - x1) * (y2 - y1)], dtype=np.float32), 'areas'
                    )

                # 指定 track_id = CSV 的 tid（关键：避免再次跟踪带来的 ID 抖动）
                ds.set_field(int(tid), 'track_id')

                # 骨架定义转换（使用数据集名字符串版本；若两者相同则相当于 no-op）
                kpts_conv = convert_keypoint_definition(
                    keypoints, pose_det_dataset_name, pose_lift_dataset_name
                )

                ds3 = PoseDataSample()
                ds3.set_field(ds.pred_instances.clone(), 'pred_instances')
                ds3.pred_instances.set_field(kpts_conv, 'keypoints')
                ds3.set_field(int(tid), 'track_id')
                # 关键：给 3D 推理准备一个“空的” gt_instances，避免 clone 时抛错
                ds3.set_field(InstanceData(), 'gt_instances')
                converted_list.append(ds3)

            # 2) 追加到时序缓存，并抽取本帧所需的时序序列
            pose_est_results_list.append(converted_list)

            pose_seq_2d = extract_pose_sequence(
                pose_est_results_list,
                frame_idx=frame_idx,
                causal=causal,
                seq_len=seq_len,
                step=seq_step
            )

            # 3) 只保留时间长度恰好等于 seq_len 的实例，其他轨迹先跳过
            def _get_T(di):
                if isinstance(di, dict):
                    x = di.get('inputs', di.get('input', None))
                    if x is not None and hasattr(x, 'shape') and len(x.shape) > 0:
                        return x.shape[0]
                    ds_ = di.get('data_samples', None)
                    return len(ds_) if ds_ is not None and hasattr(ds_, '__len__') else 0
                if hasattr(di, 'shape') and len(di.shape) > 0:
                    return di.shape[0]
                if isinstance(di, (list, tuple)):
                    return len(di)
                return 0

            ready_batches = [di for di in pose_seq_2d if _get_T(di) == seq_len]

            if len(ready_batches) > 0:
                # 4) 3D 推理（仅对“窗口就绪”的实例）
                pl_results = inference_pose_lifter_model(
                    pose_lifter,
                    ready_batches,
                    image_size=vis_img.shape[:2],
                    norm_pose_2d=(not disable_norm_pose_2d)
                )

                # 5) 官方后处理：换轴/翻符号/可选 rebase，然后写入 JSON（不改画面）
                for di, plr in zip(ready_batches, pl_results):
                    tid = int(di.get('track_id', -1)) if isinstance(di, dict) else -1
                    pred = plr.pred_instances
                    k3d = pred.keypoints
                    s3d = getattr(pred, 'keypoint_scores', None)

                    if k3d is None or len(k3d) == 0 or tid < 0:
                        continue
                    if k3d.ndim == 4:
                        k3d = np.squeeze(k3d, axis=1)
                    # [x,y,z] → [x,z,y]，并翻转 x/z；可选 rebase
                    k3d = k3d[..., [0, 2, 1]]
                    k3d[..., 0] = -k3d[..., 0]
                    k3d[..., 2] = -k3d[..., 2]
                    if not disable_rebase_keypoint:
                        k3d[..., 2] -= np.min(k3d[..., 2], axis=-1, keepdims=True)

                    results3d.setdefault(frame_idx, {})[tid] = {
                        "keypoints_3d": k3d[0].tolist(),
                        "score": (float(s3d[0].mean()) if (s3d is not None and len(s3d) > 0) else None)
                    }

        # === 输出（与原来一致） ===
        if out_path:
            writer.write(vis_img)
        if show:
            cv2.imshow("MMPose from Tracks (2D)", vis_img)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        if show_progress:
            pbar.update(1)

    # 收尾
    cap.release()
    if show_progress:
        pbar.close()
    if out_path:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    # 可选：保存 3D JSON（不影响原有视频输出）
    if use_3d and save_3d_json:
        with open(save_3d_json, "w", encoding="utf-8") as f:
            json.dump(results3d, f)
        print(f"已保存 3D 结果到：{save_3d_json}")

    print("完成。输出：", out_path if out_path else "(未保存)")


def main():
    ap = argparse.ArgumentParser()
    # ===== 原有 2D 参数（保持不变）=====
    ap.add_argument("--pose-config", required=True, help="2D Top-Down config")
    ap.add_argument("--pose-checkpoint", required=True, help="2D Top-Down checkpoint or URL")
    ap.add_argument("--video", required=True, help="输入视频路径")
    ap.add_argument("--tracks-dir", required=True, help="包含 track_*.csv 的目录")
    ap.add_argument("--out", default="", help="输出视频路径（.mp4/.avi）；留空则不保存")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kpt-thr", type=float, default=0.3)
    ap.add_argument("--expand", type=float, default=1.15, help="bbox 放大比例")
    ap.add_argument("--extra", type=int, default=4, help="bbox 额外留白像素")
    ap.add_argument("--show", action="store_true", help="是否实时显示窗口")
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--draw-bbox", action="store_true")
    ap.add_argument("--no-progress", action="store_true", help="关闭控制台进度条（默认开启）")

    # ===== 3D（全部可选；不提供就不会启用）=====
    ap.add_argument("--pose-lifter-config", default="", help="3D Pose Lifter config（可选）")
    ap.add_argument("--pose-lifter-checkpoint", default="", help="3D Pose Lifter checkpoint（可选）")
    ap.add_argument("--seq-len", type=int, default=0, help="留 0 则使用 lifter 数据集默认")
    ap.add_argument("--save-3d-json", default="", help="可选：将 3D 结果保存为 JSON 文件路径")
    ap.add_argument("--disable-rebase-keypoint", action="store_true")
    ap.add_argument("--disable-norm-pose-2d", action="store_true")

    args = ap.parse_args()

    run_video_with_tracks(
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_checkpoint,
        video_path=args.video,
        tracks_dir=args.tracks_dir,
        out_path=(args.out if args.out else None),
        device=args.device,
        kpt_score_thr=args.kpt_thr,
        expand=args.expand,
        extra=args.extra,
        show=args.show,
        radius=args.radius,
        thickness=args.thickness,
        alpha=args.alpha,
        draw_bbox=args.draw_bbox,
        show_progress=(not args.no_progress),
        # 3D
        pose_lifter_config=(args.pose_lifter_config or None),
        pose_lifter_checkpoint=(args.pose_lifter_checkpoint or None),
        seq_len=(None if args.seq_len in (None, 0) else args.seq_len),
        save_3d_json=(args.save_3d_json or None),
        disable_rebase_keypoint=args.disable_rebase_keypoint,
        disable_norm_pose_2d=args.disable_norm_pose_2d,
    )


if __name__ == "__main__":
    main()

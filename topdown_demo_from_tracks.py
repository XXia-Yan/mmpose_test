#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topdown_demo_from_tracks.py
---------------------------
用你项目里导出的 track_*.csv 作为人体框来源（而不是 mmdet），在视频上跑 MMPose 的 top-down 姿态估计。
支持 MMPose 1.x（mmengine 体系）与 0.x（mmcv 体系）。

CSV 期望格式（与你当前导出的相同字段名，按帧追加）:
    frame,x1,y1,x2,y2,score,mask,area
- 若某帧该轨迹丢失，x1..y2 为 None。
- 目录里放多份：track_0.csv, track_1.csv, ...

用法示例：
python topdown_demo_from_tracks.py ^
  --pose-config ./configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py ^
  --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth ^
  --video ./video/mabei.mp4 ^
  --tracks-dir ./mabei_yolo_tranreid_multibody_kalman ^
  --out ./video/mabei_pose_from_tracks.mp4 ^
  --show

备注：
- 若 --show 开启，会弹窗预览（按 ESC 退出）。
- --kpt-thr 控制可视化时关键点置信度阈值。
- 如果 CSV 中 score 为空，将默认 1.0。
"""
from __future__ import annotations
from tqdm.auto import tqdm
import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple, Any, Optional

import cv2
import mmcv
import mmengine
import numpy as np

# ---- MMPose 1.x 标准导入（参照官方示例）----
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

# ----------------- 实用函数 -----------------
def _expand_bbox(b: List[float], H: int, W: int, scale: float = 1.15, extra: int = 4) -> List[int]:
    """适度放大 bbox，避免裁剪肢体；返回 [x1,y1,x2,y2]（整数、已 clamp）"""
    x1, y1, x2, y2 = map(float, b)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    nw, nh = w * scale + 2 * extra, h * scale + 2 * extra
    nx1 = max(0, int(round(cx - nw / 2.0)))
    ny1 = max(0, int(round(cy - nh / 2.0)))
    nx2 = min(W, int(round(cx + nw / 2.0)))
    ny2 = min(H, int(round(cy + nh / 2.0)))
    if nx2 <= nx1:
        nx2 = min(W, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(H, ny1 + 1)
    return [nx1, ny1, nx2, ny2]


def load_tracks(tracks_dir: str) -> Tuple[Dict[int, List[Dict[str, Any]]], int]:
    """
    读取 tracks_dir 下的 track_*.csv，聚合为：
    frame_index -> [ { 'bbox':[x1,y1,x2,y2], 'score':float, 'tid':int }, ... ]
    返回 (frame_map, max_frame_idx)
    """
    csv_paths = sorted(glob.glob(os.path.join(tracks_dir, "track_*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"未在 {tracks_dir} 下找到 track_*.csv")

    frame_map: Dict[int, List[Dict[str, Any]]] = {}
    max_frame = 0
    for p in csv_paths:
        # 解析 tid
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


# ----------------- 主流程（MMPose 1.x API） -----------------
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
    show_progress: bool = True,          # <- 新增
):
    # 1) 初始化姿态模型（官方推荐：init_model as init_pose_estimator）
    pose_estimator = init_pose_estimator(
        pose_config,
        pose_checkpoint,
        device=device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
    )

    # 2) 可视化器（官方 1.x 用法：VISUALIZERS + add_datasample）
    pose_estimator.cfg.visualizer.radius = radius
    pose_estimator.cfg.visualizer.alpha = alpha
    pose_estimator.cfg.visualizer.line_width = thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # dataset_meta 来自 checkpoint
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style="mmpose")

    # 3) 读取 CSV 索引
    frame_map, _ = load_tracks(tracks_dir)

    # 4) 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=(total_frames if total_frames > 0 else None),
                unit="frame", disable=not show_progress)
    
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # 当帧的人框（扩大+裁边），组装为 Nx4 ndarray，符合 inference_topdown 的 1.x 接口
        items = frame_map.get(frame_idx, [])
        bxs = []
        tids = []
        for it in items:
            bx = _expand_bbox(it["bbox"], H, W, scale=expand, extra=extra)
            bxs.append(bx)
            tids.append(int(it.get("tid", 0)))
        if len(bxs) == 0:
            # 没有人框，直接把原帧写出/显示
            vis_img = frame.copy()
            if writer is not None:
                writer.write(vis_img)
            if show:
                cv2.imshow("MMPose from Tracks", vis_img)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
            continue

        bboxes = np.asarray(bxs, dtype=np.float32)  # [N,4], xyxy

        # 5) 姿态推理（1.x）：直接传入 Nx4 的 bboxes
        pose_results = inference_topdown(pose_estimator, frame, bboxes)
        data_samples = merge_data_samples(pose_results)  # 合并为一个 DataSample

        # 6) 画图：可视化器 add_datasample（支持画 keypoints / bbox）
        # 注意：可视化器要求 RGB，这里传 BGR 会在内部转；我们统一按官方习惯传 RGB
        rgb = mmcv.bgr2rgb(frame)
        visualizer.add_datasample(
            "result",
            rgb,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=draw_bbox,
            show=False,
            wait_time=0,
            kpt_thr=kpt_score_thr,
        )

        # 取出可视化后的图像（RGB），转换回 BGR 写视频/显示
        vis_img = visualizer.get_image()
        vis_img = mmcv.rgb2bgr(vis_img)

        # 7) 叠加你需要的 track id 标注（简单在每个框左上角写 ID）
        for bx, tid in zip(bxs, tids):
            x1, y1, x2, y2 = map(int, bx)
            cv2.putText(vis_img, f"ID{tid}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 8) 输出 / 显示
        if writer is not None:
            writer.write(vis_img)
        if show:
            cv2.imshow("MMPose from Tracks", vis_img)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
        # 更新进度条
        if show_progress:
            pbar.update(1)

    cap.release()
    if show_progress:
        pbar.close()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print("完成。输出：", out_path if out_path else "(未保存)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-config", required=True, help="MMPose config 路径")
    ap.add_argument("--pose-checkpoint", required=True, help="MMPose checkpoint 路径或 URL")
    ap.add_argument("--video", required=True, help="输入视频路径")
    ap.add_argument("--tracks-dir", required=True, help="包含 track_*.csv 的目录")
    ap.add_argument("--out", default="", help="输出视频路径（.mp4）；留空则不保存")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kpt-thr", type=float, default=0.3)
    ap.add_argument("--expand", type=float, default=1.15, help="bbox 放大比例")
    ap.add_argument("--extra", type=int, default=4, help="bbox 额外留白像素")
    ap.add_argument("--show", action="store_true", help="是否实时显示窗口")
    # 与官方可视化参数对齐
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--draw-bbox", action="store_true")

    ap.add_argument("--no-progress", action="store_true",
                help="关闭控制台进度条（默认开启）")
    ap.add_argument("--overlay-progress", action="store_true",
                    help="在视频画面左上角叠加百分比文本")
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
        show_progress=(not args.no_progress),           # <- 新增
    )


if __name__ == "__main__":
    main()
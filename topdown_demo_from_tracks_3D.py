#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topdown_demo_from_tracks_3D.py
- 只跑一次 2D（CSV bbox+tid 驱动），视频只画 2D。
- 3D：前后对称（非因果）滑动窗口；若窗口内某帧缺该 tid，
      则沿当前方向（左半向过去、右半向未来、中心位就近）继续找“更远的真实帧”补足，
      直到满足 seq_len；不做任何 padding/复制。
- 新增：2D 缓存（保存/加载），下次可跳过 2D 直接进入 3D 与可视化。
- 新增：3D 输入清洗 _sanitize_ready_batches，避免类型/字段异常导致崩溃。

示例（首次生成并缓存2D）：
python topdown_demo_from_tracks_3D.py ^
  --pose-config ./configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py ^
  --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth ^
  --pose-lifter-config ./configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py ^
  --pose-lifter-checkpoint ./checkpoints/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.pth ^
  --video ./video/mabei.mp4 ^
  --tracks-dir ./mabei_yolo_tranreid_multibody_kalman ^
  --out ./video/mabei_pose_from_tracks_3d.mp4 ^
  --seq-len 27 ^
  --save-3d-json ./video/out_3d.json ^
  --save-2d-cache ./cache/mabei_2d.json ^
  --show

下次直接复用2D：
python topdown_demo_from_tracks_3D.py  (同上参数)  --load-2d-cache ./cache/mabei_2d.json
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from tqdm.auto import tqdm
from mmengine.structures import InstanceData

# ===== 2D（MMPose 1.x）=====
from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis import inference_topdown
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, PoseDataSample

# ===== 3D（官方调用链）=====
from mmpose.apis import (
    init_model as init_model_official,  # lifter 也用 init_model
    convert_keypoint_definition,
    inference_pose_lifter_model,
)

# 官方 Compose（用 lifter 的 test pipeline）
from mmengine.dataset import Compose


# ----------------- 工具函数 -----------------
def _expand_bbox(b: List[float], H: int, W: int, scale: float = 1.15, extra: int = 4) -> List[int]:
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


def _build_pose_sample_from_cached(kpts: np.ndarray, bbox: List[float], tid: int) -> PoseDataSample:
    ds = PoseDataSample()
    inst = InstanceData()
    inst.set_field(np.asarray([bbox], dtype=np.float32), 'bboxes')
    x1, y1, x2, y2 = bbox
    inst.set_field(np.asarray([(x2 - x1) * (y2 - y1)], dtype=np.float32), 'areas')
    inst.set_field(kpts.astype(np.float32), 'keypoints')
    ds.set_field(inst, 'pred_instances')
    ds.set_field(int(tid), 'track_id')
    ds.set_field(InstanceData(), 'gt_instances')
    return ds


def _save_2d_cache(path: str,
                   frame_cache: Dict[int, Dict[int, PoseDataSample]],
                   det_dataset_name: str,
                   lift_dataset_name: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {
        "det_dataset_name": det_dataset_name,
        "lift_dataset_name": lift_dataset_name,
        "frames": {}
    }
    for fi, d in frame_cache.items():
        out["frames"][str(fi)] = {}
        for tid, ds in d.items():
            pred_np = ds.pred_instances.cpu().numpy()
            kpts = pred_np.keypoints  # [K,3]
            bbox = pred_np.bboxes[0].tolist() if hasattr(pred_np, 'bboxes') else [0, 0, 0, 0]
            area = float(pred_np.areas[0]) if hasattr(pred_np, 'areas') else 0.0
            out["frames"][str(fi)][str(tid)] = {
                "keypoints": kpts.tolist(),
                "bbox": bbox,
                "area": area
            }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[2D cache] saved to {path}")


def _load_2d_cache(path: str) -> Tuple[Dict[int, Dict[int, PoseDataSample]], str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    det_name = data.get("det_dataset_name", "")
    lift_name = data.get("lift_dataset_name", "")
    frames = data.get("frames", {})
    frame_cache: Dict[int, Dict[int, PoseDataSample]] = {}
    for sfi, d in frames.items():
        fi = int(sfi)
        frame_cache[fi] = {}
        for stid, rec in d.items():
            tid = int(stid)
            kpts = np.asarray(rec["keypoints"], dtype=np.float32)
            bbox = list(map(float, rec.get("bbox", [0, 0, 0, 0])))
            ds = _build_pose_sample_from_cached(kpts, bbox, tid)
            frame_cache[fi][tid] = ds
    print(f"[2D cache] loaded from {path} (frames={len(frame_cache)})")
    return frame_cache, det_name, lift_name


def _sanitize_ready_batches(ready_batches):
    """
    确保 ready_batches 结构符合 inference_pose_lifter_model 的预期：
      - 每个 item 是 dict，且包含 'data_samples': List[PoseDataSample]
      - 每个 PoseDataSample 都带有 gt_instances（即便是空的）
    如遇异常类型，打印并丢弃；空 batch 丢弃。
    """
    fixed = []
    dropped = 0
    for bi, di in enumerate(ready_batches):
        if not isinstance(di, dict):
            print(f"[WARN] ready_batches[{bi}] 非 dict：{type(di)} → 丢弃")
            continue
        ds_list = di.get('data_samples', None)
        if not isinstance(ds_list, list):
            print(f"[WARN] ready_batches[{bi}]['data_samples'] 非 list：{type(ds_list)} → 丢弃")
            continue

        new_list = []
        for si, ds in enumerate(ds_list):
            if not isinstance(ds, PoseDataSample):
                print(f"[WARN] batch[{bi}].data_samples[{si}] 类型异常：{type(ds)}（期望 PoseDataSample）→ 丢弃该元素")
                continue
            if not hasattr(ds, 'gt_instances'):
                ds.set_field(InstanceData(), 'gt_instances')
            new_list.append(ds)
        if len(new_list) == 0:
            dropped += 1
            continue
        di['data_samples'] = new_list
        fixed.append(di)
    if dropped > 0:
        print(f"[INFO] 清理无效 batch：丢弃 {dropped} 个")
    return fixed


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
    # ===== 2D 缓存 =====
    load_2d_cache: Optional[str] = None,
    save_2d_cache: Optional[str] = None,
    # ===== 3D 可选 =====
    pose_lifter_config: Optional[str] = None,
    pose_lifter_checkpoint: Optional[str] = None,
    seq_len: Optional[int] = None,
    seq_step: int = 1,
    save_3d_json: Optional[str] = None,
    disable_rebase_keypoint: bool = False,
    disable_norm_pose_2d: bool = False,
):
    # 1) 读 CSV + 打开视频
    frame_map, _ = load_tracks(tracks_dir)
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

    win_name = "MMPose from Tracks (2D)"
    if show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(W, 1280), min(H, 720))

    # 2) 2D 模型与可视化器
    pose_estimator = init_pose_estimator(
        pose_config,
        pose_checkpoint,
        device=device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
    )
    pose_estimator.cfg.visualizer.radius = radius
    pose_estimator.cfg.visualizer.alpha = alpha
    pose_estimator.cfg.visualizer.line_width = thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style="mmpose")

    # 3) 3D 模型与参数
    use_3d = bool(pose_lifter_config and pose_lifter_checkpoint)
    if use_3d:
        pose_lifter = init_model_official(
            pose_lifter_config, pose_lifter_checkpoint, device=device
        )
        results3d: Dict[int, Dict[int, Dict[str, Any]]] = {}
        det_dataset_name = pose_estimator.dataset_meta.get('dataset_name', None)
        lift_dataset_name = pose_lifter.dataset_meta.get('dataset_name', None)

        ds_cfg = pose_lifter.cfg.test_dataloader.dataset
        if seq_len is None or seq_len <= 0:
            seq_len = ds_cfg.get('seq_len', 1)
        if seq_step is None or seq_step <= 0:
            seq_step = ds_cfg.get('seq_step', 1)
        half = ((seq_len - 1) // 2) * seq_step
        print(f"[3D lifter(strict-extend)] dataset={pose_lifter.dataset_meta.get('dataset_name')}, "
              f"seq_len={seq_len}, step={seq_step}, causal=False, half={half}")

        # lifter 的官方 test pipeline（关键！）
        test_pipeline = Compose(ds_cfg.pipeline)

    else:
        det_dataset_name = pose_estimator.dataset_meta.get('dataset_name', None)
        lift_dataset_name = det_dataset_name
        test_pipeline = None  # 用不到

    # 4) 2D 缓存：优先加载，否则在线推理
    frame_cache: Dict[int, Dict[int, PoseDataSample]] = {}
    use_cached_2d = False
    if load_2d_cache and os.path.isfile(load_2d_cache):
        frame_cache, _, _ = _load_2d_cache(load_2d_cache)
        use_cached_2d = True

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        items = frame_map.get(frame_idx, [])
        bxs, tids = [], []
        for it in items:
            bx = _expand_bbox(it["bbox"], H, W, scale=expand, extra=extra)
            bxs.append(bx)
            tids.append(int(it.get("tid", 0)))

        pose_results: List[PoseDataSample] = []

        if use_cached_2d:
            if frame_idx in frame_cache:
                for tid in sorted(frame_cache[frame_idx].keys()):
                    pose_results.append(frame_cache[frame_idx][tid])
            if len(pose_results) > 0:
                data_samples = merge_data_samples(pose_results)
                rgb = mmcv.bgr2rgb(frame)
                visualizer.add_datasample(
                    "result", rgb, data_sample=data_samples,
                    draw_gt=False, draw_heatmap=False, draw_bbox=draw_bbox,
                    show=False, wait_time=0, kpt_thr=kpt_score_thr,
                )
                vis_img = mmcv.rgb2bgr(visualizer.get_image())
            else:
                vis_img = frame.copy()

            for bx, tid in zip(bxs, tids):
                x1, y1, x2, y2 = map(int, bx)
                cv2.putText(vis_img, f"ID{tid}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        else:
            # 在线 2D 推理
            if len(bxs) == 0:
                vis_img = frame.copy()
            else:
                bboxes = np.asarray(bxs, dtype=np.float32)
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

                for bx, tid in zip(bxs, tids):
                    x1, y1, x2, y2 = map(int, bx)
                    cv2.putText(vis_img, f"ID{tid}", (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 把本帧 2D 转到 lifter 骨架并放入 frame_cache（供 3D 与缓存）
            if use_3d and len(pose_results) > 0:
                frame_cache.setdefault(frame_idx, {})
                for tid, bx, ds in zip(tids, bxs, pose_results):
                    pred_np = ds.pred_instances.cpu().numpy()
                    keypoints = pred_np.keypoints  # [K,3]
                    if not hasattr(ds.pred_instances, 'bboxes'):
                        ds.pred_instances.set_field(np.asarray([bx], dtype=np.float32), 'bboxes')
                    if not hasattr(ds.pred_instances, 'areas'):
                        x1, y1, x2, y2 = bx
                        ds.pred_instances.set_field(
                            np.asarray([(x2 - x1) * (y2 - y1)], dtype=np.float32), 'areas'
                        )
                    ds.set_field(int(tid), 'track_id')

                    kpts_conv = convert_keypoint_definition(
                        keypoints, det_dataset_name, lift_dataset_name
                    )
                    ds3 = PoseDataSample()
                    ds3.set_field(ds.pred_instances.clone(), 'pred_instances')
                    ds3.pred_instances.set_field(kpts_conv, 'keypoints')
                    ds3.set_field(int(tid), 'track_id')
                    ds3.set_field(InstanceData(), 'gt_instances')
                    frame_cache[frame_idx][int(tid)] = ds3

        if out_path:
            writer.write(vis_img)
        if show:
            cv2.imshow(win_name, vis_img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

        if show_progress:
            pbar.update(1)

    # 收尾视频输出
    cap.release()
    if show_progress:
        pbar.close()
    if out_path:
        writer.release()
    if show:
        cv2.destroyWindow(win_name)

    # 保存 2D 缓存（保存的是已转到 3D 骨架定义的 2D keypoints）
    if save_2d_cache and use_3d and len(frame_cache) > 0:
        _save_2d_cache(save_2d_cache, frame_cache, det_dataset_name, lift_dataset_name)

    # —— 离线构造“真实帧扩展”的对称窗口并做 3D ——
    if use_3d:
        total = max(frame_cache.keys()) if len(frame_cache) > 0 else 0
        half = ((seq_len - 1) // 2) * seq_step
        start_center = 1 + half
        end_center = total - half

        # present[fi] = set([tid,...])：CSV中该帧出现的 tid 加速查询
        present: Dict[int, set] = {}
        for fi, lst in frame_map.items():
            present[fi] = set(int(it['tid']) for it in lst)

        def _find_real_frame_for_offset(tid: int, target_fi: int, direction: int) -> Optional[int]:
            fi = target_fi
            if 1 <= fi <= total and (fi in present) and (tid in present[fi]):
                return fi
            fi += direction
            while 1 <= fi <= total:
                if (fi in present) and (tid in present[fi]):
                    return fi
                fi += direction
            return None

        results3d: Dict[int, Dict[int, Dict[str, Any]]] = {}

        if start_center <= end_center:
            for center_idx in range(start_center, end_center + 1):
                ready_batches: List[Dict[str, Any]] = []
                tids_here = list(frame_cache.get(center_idx, {}).keys())
                offsets = list(range(-half, half + 1, seq_step))  # 对称窗口

                for tid in tids_here:
                    used_frames = set()
                    seq_frame_indices: List[int] = []
                    ok_all = True

                    for off in offsets:
                        target_fi = center_idx + off
                        direction = -1 if off < 0 else (+1 if off > 0 else 0)

                        if direction == 0:
                            chosen = None
                            if (target_fi in present) and (tid in present[target_fi]):
                                chosen = target_fi
                            else:
                                back = _find_real_frame_for_offset(tid, target_fi, -1)
                                fwd = _find_real_frame_for_offset(tid, target_fi, +1)
                                if back is None and fwd is None:
                                    ok_all = False
                                else:
                                    if back is None:
                                        chosen = fwd
                                    elif fwd is None:
                                        chosen = back
                                    else:
                                        if abs(back - target_fi) <= abs(fwd - target_fi):
                                            chosen = back
                                        else:
                                            chosen = fwd
                            if not ok_all:
                                break
                        else:
                            chosen = _find_real_frame_for_offset(tid, target_fi, direction)
                            if chosen is None:
                                ok_all = False
                                break

                        if chosen in used_frames:
                            step_dir = direction if direction != 0 else +1
                            next_fi = chosen + step_dir
                            found = None
                            while 1 <= next_fi <= total:
                                if (next_fi in present) and (tid in present[next_fi]) and (next_fi not in used_frames):
                                    found = next_fi
                                    break
                                next_fi += step_dir
                            if found is None:
                                ok_all = False
                                break
                            chosen = found

                        used_frames.add(chosen)
                        seq_frame_indices.append(chosen)

                    if not ok_all:
                        continue

                    seq_frame_indices.sort()
                    if len(seq_frame_indices) != len(offsets):
                        continue

                    # 收集 PoseDataSample 序列
                    seq_list: List[PoseDataSample] = []
                    make_ok = True
                    for fi in seq_frame_indices:
                        ds3 = frame_cache.get(fi, {}).get(tid, None)
                        if not isinstance(ds3, PoseDataSample):
                            print(f"[WARN] frame={fi}, tid={tid} 不是 PoseDataSample：{type(ds3)}，跳过")
                            make_ok = False
                            break
                        if not hasattr(ds3, 'gt_instances'):
                            ds3.set_field(InstanceData(), 'gt_instances')
                        seq_list.append(ds3)
                    if not make_ok or len(seq_list) != len(offsets):
                        continue

                    # ★★★ 官方风格：先经过 lifter 的 test pipeline，再进入 3D ★★★
                    #   pipeline 会返回 {'inputs': ..., 'data_samples': ...} 的标准结构
                    data_info = dict(data_samples=seq_list)  # inputs 留给 pipeline 构造
                    processed = test_pipeline(data_info)
                    # 附带 track_id，便于后续结果落盘
                    processed['track_id'] = int(tid)
                    ready_batches.append(processed)

                if len(ready_batches) == 0:
                    continue

                # 保险：清洗（类型/字段）——正常应该是空操作
                ready_batches = _sanitize_ready_batches(ready_batches)

                # 3D 推理
                pl_results = inference_pose_lifter_model(
                    pose_lifter,
                    ready_batches,
                    image_size=(H, W),
                    norm_pose_2d=(not disable_norm_pose_2d)
                )

                # 后处理 & JSON 落盘（键为中心帧）
                for di, plr in zip(ready_batches, pl_results):
                    tid = int(di.get('track_id', -1))
                    pred = plr.pred_instances
                    k3d = pred.keypoints
                    s3d = getattr(pred, 'keypoint_scores', None)
                    if k3d is None or len(k3d) == 0 or tid < 0:
                        continue
                    if k3d.ndim == 4:
                        k3d = np.squeeze(k3d, axis=1)
                    k3d = k3d[..., [0, 2, 1]]   # x,y,z -> x,z,y
                    k3d[..., 0] = -k3d[..., 0]  # flip x
                    k3d[..., 2] = -k3d[..., 2]  # flip z
                    if not disable_rebase_keypoint:
                        k3d[..., 2] -= np.min(k3d[..., 2], axis=-1, keepdims=True)
                    results3d.setdefault(center_idx, {})[tid] = {
                        "keypoints_3d": k3d[0].tolist(),
                        "score": (float(s3d[0].mean()) if (s3d is not None and len(s3d) > 0) else None)
                    }

        if save_3d_json:
            with open(save_3d_json, "w", encoding="utf-8") as f:
                json.dump(results3d, f)
            print(f"已保存 3D 结果到：{save_3d_json}")

    print("完成。输出：", out_path if out_path else "(未保存)")


def main():
    ap = argparse.ArgumentParser()
    # ===== 2D =====
    ap.add_argument("--pose-config", required=True)
    ap.add_argument("--pose-checkpoint", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kpt-thr", type=float, default=0.3)
    ap.add_argument("--expand", type=float, default=1.15)
    ap.add_argument("--extra", type=int, default=4)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--draw-bbox", action="store_true")
    ap.add_argument("--no-progress", action="store_true")

    # ===== 2D 缓存 =====
    ap.add_argument("--load-2d-cache", default="", help="加载已保存的2D关键点缓存（JSON）")
    ap.add_argument("--save-2d-cache", default="", help="将本次2D关键点缓存保存到（JSON）")

    # ===== 3D =====
    ap.add_argument("--pose-lifter-config", default="")
    ap.add_argument("--pose-lifter-checkpoint", default="")
    ap.add_argument("--seq-len", type=int, default=0)
    ap.add_argument("--seq-step", type=int, default=1)
    ap.add_argument("--save-3d-json", default="")
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
        # 2D 缓存
        load_2d_cache=(args.load_2d_cache or None),
        save_2d_cache=(args.save_2d_cache or None),
        # 3D
        pose_lifter_config=(args.pose_lifter_config or None),
        pose_lifter_checkpoint=(args.pose_lifter_checkpoint or None),
        seq_len=(None if args.seq_len in (None, 0) else args.seq_len),
        seq_step=(args.seq_step if args.seq_step > 0 else 1),
        save_3d_json=(args.save_3d_json or None),
        disable_rebase_keypoint=args.disable_rebase_keypoint,
        disable_norm_pose_2d=args.disable_norm_pose_2d,
    )


if __name__ == "__main__":
    main()

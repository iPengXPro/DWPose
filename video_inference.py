#!/usr/bin/env python
"""
DWPose 视频推理脚本
处理视频，绘制姿态骨架，保存结果视频

Usage:
    python video_inference.py --input video.mp4 --output result.mp4
    python video_inference.py -i video.mp4 -o result.mp4 --skip 2
"""

import os
import sys
import cv2
import numpy as np
import math
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "DWPose-models"

# 导入推理函数
dwpose_path = BASE_DIR / "ControlNet-v1-1-nightly" / "annotator" / "dwpose"
sys.path.insert(0, str(dwpose_path))

import onnxruntime as ort
from onnxdet import inference_detector
from onnxpose import inference_pose


# ========== 绘制函数 ==========

def draw_bodypose(canvas, keypoints, scores, score_thr=0.3):
    """绘制身体骨架"""
    H, W = canvas.shape[:2]

    limbSeq = [
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
        [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 255, 255]]

    stickwidth = 4

    for person_idx in range(len(keypoints)):
        kps = keypoints[person_idx]
        scs = scores[person_idx]

        for i in range(len(limbSeq)):
            p1_idx = limbSeq[i][0]
            p2_idx = limbSeq[i][1]

            if p1_idx >= len(kps) or p2_idx >= len(kps):
                continue
            if scs[p1_idx] < score_thr or scs[p2_idx] < score_thr:
                continue

            x1, y1 = kps[p1_idx]
            x2, y2 = kps[p2_idx]

            if x1 < 0 or x2 < 0:
                continue

            Y = [x1, x2]
            X = [y1, y2]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for person_idx in range(len(keypoints)):
        kps = keypoints[person_idx]
        scs = scores[person_idx]

        for i in range(18):
            if i >= len(kps):
                break
            if scs[i] < score_thr:
                continue
            x, y = int(kps[i][0]), int(kps[i][1])
            if x >= 0 and y >= 0:
                cv2.circle(canvas, (x, y), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, keypoints, scores, score_thr=0.3, is_left=True):
    """绘制手部骨架"""
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
             [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
             [0, 17], [17, 18], [18, 19], [19, 20]]

    import colorsys

    for person_idx in range(len(keypoints)):
        kps = keypoints[person_idx]
        scs = scores[person_idx]

        if is_left:
            hand_kps = kps[92:113]
            hand_scs = scs[92:113]
        else:
            hand_kps = kps[113:133]
            hand_scs = scs[113:133]

        for ie, e in enumerate(edges):
            if e[0] >= len(hand_kps) or e[1] >= len(hand_kps):
                continue

            x1, y1 = hand_kps[e[0]]
            x2, y2 = hand_kps[e[1]]
            s1, s2 = hand_scs[e[0]], hand_scs[e[1]]

            if s1 < score_thr or s2 < score_thr:
                continue
            if x1 < 0 or x2 < 0:
                continue

            hue = ie / len(edges)
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))

            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), bgr, thickness=2)

    return canvas


def draw_facepose(canvas, keypoints, scores, score_thr=0.3):
    """绘制面部关键点"""
    H, W = canvas.shape[:2]

    for person_idx in range(len(keypoints)):
        kps = keypoints[person_idx]
        scs = scores[person_idx]

        face_kps = kps[24:92]
        face_scs = scs[24:92]

        for kpt, score in zip(face_kps, face_scs):
            if score < score_thr:
                continue
            x, y = int(kpt[0]), int(kpt[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)

    return canvas


def draw_pose_on_image(img, keypoints, scores, score_thr=0.3):
    """在图像上绘制完整姿态"""
    canvas = img.copy()
    canvas = draw_bodypose(canvas, keypoints, scores, score_thr)
    canvas = draw_handpose(canvas, keypoints, scores, score_thr, is_left=True)
    canvas = draw_handpose(canvas, keypoints, scores, score_thr, is_left=False)
    canvas = draw_facepose(canvas, keypoints, scores, score_thr)
    return canvas


def remap_keypoints(keypoints, scores):
    """关键点重映射：mmpose -> OpenPose"""
    keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)

    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    neck[:, 2:4] = np.logical_and(
        keypoints_info[:, 5, 2:4] > 0.3,
        keypoints_info[:, 6, 2:4] > 0.3).astype(int)

    new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]

    return new_keypoints_info[..., :2], new_keypoints_info[..., 2]


def load_models():
    """加载模型"""
    det_model = MODEL_DIR / "yolox_l.onnx"
    pose_model = MODEL_DIR / "dw-ll_ucoco_384.onnx"

    if not det_model.exists():
        print(f"Error: {det_model} not found")
        sys.exit(1)
    if not pose_model.exists():
        print(f"Error: {pose_model} not found")
        sys.exit(1)

    providers = ['CPUExecutionProvider']
    print("Loading models...")
    print(f"  Detection: {det_model.name}")
    print(f"  Pose: {pose_model.name}")

    session_det = ort.InferenceSession(str(det_model), providers=providers)
    session_pose = ort.InferenceSession(str(pose_model), providers=providers)

    return session_det, session_pose


def process_video(input_path, output_path, session_det, session_pose, skip_frames=1, show_progress=True):
    """处理视频"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return False

    # 视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info:")
    print(f"  FPS: {fps}")
    print(f"  Size: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    print(f"  Skip frames: {skip_frames}")
    print()

    # 输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps / skip_frames, (width, height))
    if not out.isOpened():
        print(f"Error: Cannot create output video {output_path}")
        cap.release()
        return False

    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 跳帧处理
        if frame_idx % skip_frames != 0:
            continue

        # 推理
        dets = inference_detector(session_det, frame)

        if len(dets) > 0:
            keypoints, scores = inference_pose(session_pose, dets, frame)
            keypoints, scores = remap_keypoints(keypoints, scores)
            frame = draw_pose_on_image(frame, keypoints, scores)

        out.write(frame)
        processed_frames += 1

        # 进度显示
        if show_progress and processed_frames % 30 == 0:
            progress = frame_idx / total_frames * 100
            print(f"  Progress: {progress:.1f}% ({frame_idx}/{total_frames})")

    cap.release()
    out.release()

    print(f"\nProcessed {processed_frames} frames")
    print(f"Output saved: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="DWPose Video Inference")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("--skip", type=int, default=1, help="Skip frames (1=process all, 2=skip half)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input video not found: {input_path}")
        sys.exit(1)

    # 加载模型
    session_det, session_pose = load_models()
    print()

    # 处理视频
    print("="*60)
    print(f"Processing: {input_path.name}")
    print("="*60)

    success = process_video(
        input_path, output_path,
        session_det, session_pose,
        skip_frames=args.skip,
        show_progress=not args.no_progress
    )

    if success:
        print("\nDone!")
    else:
        print("\nFailed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
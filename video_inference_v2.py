#!/usr/bin/env python
"""
DWPose 视频推理脚本 - 使用 ControlNet 原生代码
完全参照 ControlNet-v1-1-nightly/annotator/dwpose 的实现

Usage:
    python video_inference_v2.py --input video.mp4 --output result.mp4
    python video_inference_v2.py -i video.mp4 -o result.mp4 --cpu
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

# 设置环境
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = Path(__file__).parent.resolve()
CONTROLNET_PATH = BASE_DIR / "ControlNet-v1-1-nightly" / "annotator"
sys.path.insert(0, str(CONTROLNET_PATH))

import onnxruntime as ort


# ========== 模型路径 ==========
MODEL_DIR = BASE_DIR / "DWPose-models"
DET_MODEL = MODEL_DIR / "yolox_l.onnx"
POSE_MODEL = MODEL_DIR / "dw-ll_ucoco_384.onnx"


# ========== 推理类 (参照 wholebody.py) ==========

class Wholebody:
    """参照 ControlNet wholebody.py"""

    def __init__(self, use_gpu=True):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

        print(f"Loading models...")
        print(f"  Detection: {DET_MODEL.name}")
        print(f"  Pose: {POSE_MODEL.name}")
        print(f"  Providers: {providers}")

        self.session_det = ort.InferenceSession(str(DET_MODEL), providers=providers)
        self.session_pose = ort.InferenceSession(str(POSE_MODEL), providers=providers)

        # 显示实际设备
        print(f"  Actual: det={self.session_det.get_providers()[0]}, pose={self.session_pose.get_providers()[0]}")

    def __call__(self, oriImg):
        from dwpose.onnxdet import inference_detector
        from dwpose.onnxpose import inference_pose

        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]

        return keypoints, scores


# ========== 绘制函数 (参照 util.py) ==========

import math
import matplotlib

eps = 0.01


def draw_bodypose(canvas, candidate, subset):
    """参照 util.py draw_bodypose"""
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks):
    """参照 util.py draw_handpose"""
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2),
                        matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks):
    """参照 util.py draw_facepose"""
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


# ========== 主处理函数 (参照 __init__.py) ==========

def draw_pose(pose, H, W):
    """参照 __init__.py draw_pose"""
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = draw_bodypose(canvas, candidate, subset)
    canvas = draw_handpose(canvas, hands)
    canvas = draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    """参照 __init__.py DWposeDetector"""

    def __init__(self, use_gpu=True):
        self.pose_estimation = Wholebody(use_gpu=use_gpu)

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape

        candidate, subset = self.pose_estimation(oriImg)
        nums, keys, locs = candidate.shape
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        body = candidate[:,:18].copy()
        body = body.reshape(nums*18, locs)
        score = subset[:,:18]
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > 0.3:
                    score[i][j] = int(18*i+j)
                else:
                    score[i][j] = -1

        un_visible = subset<0.3
        candidate[un_visible] = -1

        foot = candidate[:,18:24]

        faces = candidate[:,24:92]

        hands = candidate[:,92:113]
        hands = np.vstack([hands, candidate[:,113:]])

        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)

        return draw_pose(pose, H, W)


# ========== 视频处理 ==========

def process_video(input_path, output_path, detector, skip_frames=1, show_progress=True):
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

        # 使用 DWposeDetector 处理
        pose_img = detector(frame)

        # 叠加原图
        result = cv2.addWeighted(frame, 0.6, pose_img, 0.8, 0)

        out.write(result)
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
    parser = argparse.ArgumentParser(description="DWPose Video Inference (ControlNet version)")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("--skip", type=int, default=1, help="Skip frames (1=process all, 2=skip half)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode (default: GPU if available)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input video not found: {input_path}")
        sys.exit(1)

    if not DET_MODEL.exists():
        print(f"Error: Detection model not found: {DET_MODEL}")
        sys.exit(1)
    if not POSE_MODEL.exists():
        print(f"Error: Pose model not found: {POSE_MODEL}")
        sys.exit(1)

    # 创建检测器
    use_gpu = not args.cpu
    detector = DWposeDetector(use_gpu=use_gpu)
    print()

    # 处理视频
    print("="*60)
    print(f"Processing: {input_path.name}")
    print("="*60)

    success = process_video(
        input_path, output_path,
        detector,
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
#!/usr/bin/env python
"""
DWPose 视频推理脚本 - 直接调用 ControlNet 代码

Usage:
    python video_inference_v2.py --input video.mp4 --output result.mp4
    python video_inference_v2.py -i video.mp4 -o result.mp4 --cpu
"""

import os
import sys
import cv2
import argparse
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = Path(__file__).parent.resolve()
CONTROLNET_PATH = BASE_DIR / "ControlNet-v1-1-nightly"
sys.path.insert(0, str(CONTROLNET_PATH))

# 模型路径
MODEL_DIR = BASE_DIR / "DWPose-models"
DET_MODEL = str(MODEL_DIR / "yolox_l.onnx")
POSE_MODEL = str(MODEL_DIR / "dw-ll_ucoco_384.onnx")

# 直接使用 ControlNet 的 DWPose
from annotator.dwpose import DWposeDetector


def process_video(input_path, output_path, detector, skip_frames=1, show_progress=True):
    """处理视频，按 ESC 或 Q 键提前终止"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height}, {fps:.1f}fps, {total_frames} frames")
    print(f"Skip: every {skip_frames} frame(s)")
    print("Press ESC or Q to stop early\n")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps / skip_frames, (width, height))
    if not out.isOpened():
        print(f"Error: Cannot create output video")
        cap.release()
        return False

    frame_idx = 0
    processed = 0
    stopped = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % skip_frames != 0:
            continue

        # 直接调用 DWposeDetector
        pose_img = detector(frame)

        # 叠加原图
        result = cv2.addWeighted(frame, 0.6, pose_img, 0.8, 0)
        out.write(result)
        processed += 1

        # 显示预览窗口
        cv2.imshow('DWPose Preview', cv2.resize(result, (960, 540)) if width > 960 else result)

        # 检测按键 (ESC=27, q/Q)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or key == ord('Q'):
            print(f"\n[Stopped by user at frame {frame_idx}]")
            stopped = True
            break

        if show_progress and processed % 30 == 0:
            print(f"  Progress: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nProcessed {processed} frames -> {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="DWPose Video Inference")
    parser.add_argument("-i", "--input", required=True, help="Input video")
    parser.add_argument("-o", "--output", required=True, help="Output video")
    parser.add_argument("--skip", type=int, default=1, help="Skip frames")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    # 检查模型
    if not Path(DET_MODEL).exists():
        print(f"Error: {DET_MODEL} not found")
        sys.exit(1)
    if not Path(POSE_MODEL).exists():
        print(f"Error: {POSE_MODEL} not found")
        sys.exit(1)

    print("="*60)
    print(f"Processing: {input_path.name}")
    print("="*60)

    # 使用 DWposeDetector，传入模型路径
    detector = DWposeDetector(det_model=DET_MODEL, pose_model=POSE_MODEL)

    process_video(
        input_path, args.output,
        detector,
        skip_frames=args.skip
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
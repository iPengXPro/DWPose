#!/usr/bin/env python
"""
DWPose Auto Labeling Tool
Generate YOLO Pose format labels for training.

Usage:
    python auto_label.py --images /path/to/images --output /path/to/labels --detector yolox
    python auto_label.py -i images/ -o labels/ -d yolo26
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

# Add DWPose path
BASE_DIR = Path(__file__).parent.resolve()
DWPose_PATH = BASE_DIR / "ControlNet-v1-1-nightly" / "annotator" / "dwpose"
sys.path.insert(0, str(DWPose_PATH))

import onnxruntime as ort
from onnxdet import inference_detector
from onnxpose import inference_pose


def load_models(detector_type="yolox"):
    """Load detection and pose models."""
    model_dir = BASE_DIR / "DWPose-models"

    # Select detection model
    if detector_type == "yolov8":
        det_model_path = BASE_DIR / "yolov8n.onnx"
    elif detector_type == "yolo26":
        det_model_path = BASE_DIR / "yolo26s.onnx"
    else:  # default yolox
        det_model_path = model_dir / "yolox_l.onnx"

    pose_model_path = model_dir / "dw-ll_ucoco_384.onnx"

    # Check files exist
    if not det_model_path.exists():
        print(f"Error: Detection model not found: {det_model_path}")
        sys.exit(1)
    if not pose_model_path.exists():
        print(f"Error: Pose model not found: {pose_model_path}")
        sys.exit(1)

    # Use CPU to avoid GPU compatibility issues
    providers = ['CPUExecutionProvider']

    print(f"Loading detection model: {det_model_path.name}")
    det_session = ort.InferenceSession(str(det_model_path), providers=providers)

    print(f"Loading pose model: {pose_model_path.name}")
    pose_session = ort.InferenceSession(str(pose_model_path), providers=providers)

    return det_session, pose_session, detector_type


def detect_yolox(session, img):
    """YOLOX detection (from onnxdet.py)."""
    from onnxdet import inference_detector
    return inference_detector(session, img)


def detect_yolov8(session, img, conf_thr=0.25, nms_thr=0.45, final_thr=0.3):
    """YOLOv8 detection."""
    input_h, input_w = 640, 640

    # Letterbox resize
    h, w = img.shape[:2]
    ratio = min(input_h / h, input_w / w)
    new_h, new_w = int(h * ratio), int(w * ratio)

    padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    resized = cv2.resize(img, (new_w, new_h))
    padded[:new_h, :new_w] = resized

    # HWC -> CHW, BGR -> RGB, normalize
    input_tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = input_tensor[np.newaxis, ...]

    # Inference
    output = session.run(None, {'images': input_tensor})[0]  # [1, 84, 8400]
    output = output[0].T  # [8400, 84]

    boxes = output[:, :4]  # cx, cy, w, h
    scores = output[:, 4:]

    # Filter by confidence and class 0 (person)
    max_scores = scores.max(axis=1)
    max_classes = scores.argmax(axis=1)

    mask = (max_scores > conf_thr) & (max_classes == 0)
    filtered_boxes = boxes[mask]
    filtered_scores = max_scores[mask]

    if len(filtered_boxes) == 0:
        return np.array([])

    # NMS
    x1 = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
    y1 = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
    x2 = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
    y2 = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)

    order = filtered_scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        if filtered_scores[i] < final_thr:
            break
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        iw = np.maximum(0, xx2 - xx1)
        ih = np.maximum(0, yy2 - yy1)
        inter = iw * ih
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= nms_thr)[0]
        order = order[inds + 1]

    # Convert to xyxy and scale back
    result = []
    for idx in keep:
        cx, cy, bw, bh = filtered_boxes[idx]
        bx1 = (cx - bw/2) / ratio
        by1 = (cy - bh/2) / ratio
        bx2 = (cx + bw/2) / ratio
        by2 = (cy + bh/2) / ratio
        result.append([bx1, by1, bx2, by2])

    return np.array(result) if result else np.array([])


def detect_yolo26(session, img, conf_thr=0.25):
    """YOLO26 end-to-end detection."""
    input_h, input_w = 640, 640

    # Letterbox resize with padding
    h, w = img.shape[:2]
    r = min(input_h / h, input_w / w)
    new_h, new_w = int(h * r), int(w * r)

    pad_x = (input_w - new_w) / 2
    pad_y = (input_h - new_h) / 2

    padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    resized = cv2.resize(img, (new_w, new_h))
    padded[int(pad_y):int(pad_y)+new_h, int(pad_x):int(pad_x)+new_w] = resized

    # HWC -> CHW, BGR -> RGB, normalize
    input_tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = input_tensor[np.newaxis, ...]

    # Inference - output [1, 300, 6]: x1, y1, x2, y2, conf, class
    output = session.run(None, {'images': input_tensor})[0][0]  # [300, 6]

    result = []
    for det in output:
        x1, y1, x2, y2, score, class_id = det
        if score < conf_thr or int(class_id) != 0:
            continue
        # Scale back to original image
        orig_x1 = (x1 - pad_x) / r
        orig_y1 = (y1 - pad_y) / r
        orig_x2 = (x2 - pad_x) / r
        orig_y2 = (y2 - pad_y) / r
        result.append([orig_x1, orig_y1, orig_x2, orig_y2])

    return np.array(result) if result else np.array([])


def process_image(img, det_session, pose_session, detector_type):
    """Run detection and pose estimation on an image."""
    # Detection
    if detector_type == "yolov8":
        boxes = detect_yolov8(det_session, img)
    elif detector_type == "yolo26":
        boxes = detect_yolo26(det_session, img)
    else:
        boxes = detect_yolox(det_session, img)

    if len(boxes) == 0:
        return [], []

    # Pose estimation
    keypoints, scores = inference_pose(pose_session, boxes, img)

    return boxes, keypoints, scores


def convert_to_yolo_format(img_shape, boxes, keypoints, scores):
    """
    Convert detection and pose results to YOLO Pose format.

    YOLO Pose format per line:
    class_id cx cy w h kp0_x kp0_y kp0_v kp1_x kp1_y kp1_v ...

    kp_v: 0=not visible, 1=occluded, 2=visible
    """
    h, w = img_shape[:2]
    lines = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        # Convert to YOLO format (normalized center + width/height)
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        # Clamp values to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        bw = max(0, min(1, bw))
        bh = max(0, min(1, bh))

        # Process keypoints
        kps = keypoints[i]  # [134, 2]
        scs = scores[i]      # [134]

        kp_strs = []
        for j in range(len(kps)):
            kx = kps[j, 0] / w
            ky = kps[j, 1] / h

            # Visibility: score > 0.3 = visible(2), score > 0 = occluded(1), else not visible(0)
            if scs[j] > 0.3:
                kv = 2
            elif scs[j] > 0:
                kv = 1
            else:
                kv = 0
                kx, ky = 0, 0  # YOLO convention: invisible keypoints at (0,0)

            # Clamp
            kx = max(0, min(1, kx))
            ky = max(0, min(1, ky))

            kp_strs.append(f"{kx:.6f} {ky:.6f} {kv}")

        # Build line: class_id cx cy w h kp0_x kp0_y kp0_v ...
        line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kp_strs)
        lines.append(line)

    return lines


def main():
    parser = argparse.ArgumentParser(description="DWPose Auto Labeling Tool")
    parser.add_argument("-i", "--images", required=True, help="Input images directory")
    parser.add_argument("-o", "--output", required=True, help="Output labels directory")
    parser.add_argument("-d", "--detector", choices=["yolox", "yolov8", "yolo26"],
                        default="yolox", help="Detection model type")
    parser.add_argument("--save-vis", action="store_true", help="Save visualization images")
    args = parser.parse_args()

    input_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_vis:
        vis_dir = output_dir.parent / (output_dir.name + "_vis")
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Collect images (only from specified directory, not subdirectories)
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_exts:
        for f in input_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ext:
                image_files.append(f)
            elif f.is_file() and f.suffix.lower() == ext.upper():
                image_files.append(f)

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in image_files:
        if f.name not in seen:
            seen.add(f.name)
            unique_files.append(f)
    image_files = unique_files

    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images")
    print(f"Detector: {args.detector}")
    print(f"Output: {output_dir}")
    print()

    # Load models
    det_session, pose_session, detector_type = load_models(args.detector)
    print()

    # Process images
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Processing: {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Error: Cannot read image")
            continue

        # Run detection and pose
        result = process_image(img, det_session, pose_session, detector_type)

        if len(result) == 0 or len(result[0]) == 0:
            print(f"  No persons detected")
            # Write empty label file
            label_path = output_dir / (img_path.stem + ".txt")
            label_path.write_text("")
            continue

        boxes, keypoints, scores = result
        print(f"  Detected {len(boxes)} persons, {keypoints.shape[1]} keypoints each")

        # Convert to YOLO format
        lines = convert_to_yolo_format(img.shape, boxes, keypoints, scores)

        # Save label file
        label_path = output_dir / (img_path.stem + ".txt")
        label_path.write_text("\n".join(lines))

        # Save visualization if requested
        if args.save_vis:
            vis_img = img.copy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(str(vis_dir / img_path.name), vis_img)

    print()
    print("=" * 60)
    print("Done!")
    print(f"Labels saved to: {output_dir}")
    if args.save_vis:
        print(f"Visualizations saved to: {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
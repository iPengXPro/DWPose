#!/usr/bin/env python
"""
DWPose Auto Labeling Tool - Crop Local Images
Generate YOLO Pose format labels for training with cropped person images.

Usage:
    python auto_label.py --images /path/to/images --output /path/to/output --detector yolox
    python auto_label.py -i images/ -o output/ -d yolo26

Output structure:
    output/
        images/     - cropped person images
        labels/     - YOLO pose labels (133 keypoints)
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
        return None, [], []

    # Pose estimation
    keypoints, scores = inference_pose(pose_session, boxes, img)

    return boxes, keypoints, scores


def crop_and_normalize_keypoints(img, box, keypoints, scores, padding=0.2):
    """
    Crop person region and normalize keypoints to local image coordinates.

    Args:
        img: original image
        box: [x1, y1, x2, y2] detection box
        keypoints: [133, 2] keypoints in original image coordinates
        scores: [133] keypoint scores
        padding: extra padding ratio for crop box

    Returns:
        crop_img: cropped person image
        crop_box: [x1, y1, x2, y2] actual crop box (after padding and clamping)
        local_keypoints: [133, 2] keypoints in local image coordinates
        local_scores: [133] scores (unchanged)
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box

    # Add padding
    bw = x2 - x1
    bh = y2 - y1
    pad_w = bw * padding
    pad_h = bh * padding

    crop_x1 = max(0, int(x1 - pad_w))
    crop_y1 = max(0, int(y1 - pad_h))
    crop_x2 = min(w, int(x2 + pad_w))
    crop_y2 = min(h, int(y2 + pad_h))

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    # Crop image
    crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # Convert keypoints to local coordinates
    local_keypoints = keypoints.copy()
    local_keypoints[:, 0] = keypoints[:, 0] - crop_x1  # x offset
    local_keypoints[:, 1] = keypoints[:, 1] - crop_y1  # y offset

    # Handle keypoints outside crop region
    for i in range(len(local_keypoints)):
        kx, ky = local_keypoints[i]
        if kx < 0 or kx > crop_w or ky < 0 or ky > crop_h:
            if scores[i] > 0.3:  # Was visible but now outside crop
                local_keypoints[i] = [0, 0]
                scores[i] = 0  # Mark as not visible

    return crop_img, [crop_x1, crop_y1, crop_x2, crop_y2], local_keypoints, scores


def convert_to_yolo_format_local(crop_shape, keypoints, scores):
    """
    Convert keypoints to YOLO Pose format for local cropped image.

    YOLO Pose format per line:
    class_id cx cy w h kp0_x kp0_y kp0_v kp1_x kp1_y kp1_v ...

    For cropped person image, bbox is the entire image: cx=0.5, cy=0.5, w=1, h=1

    kp_v: 0=not visible, 1=occluded, 2=visible
    """
    h, w = crop_shape[:2]

    # Bbox is entire image (normalized)
    cx, cy, bw, bh = 0.5, 0.5, 1.0, 1.0

    kp_strs = []
    for j in range(len(keypoints)):
        kx = keypoints[j, 0] / w  # Normalize to [0, 1]
        ky = keypoints[j, 1] / h

        # Visibility
        if scores[j] > 0.3:
            kv = 2
        elif scores[j] > 0:
            kv = 1
        else:
            kv = 0
            kx, ky = 0, 0

        # Clamp
        kx = max(0, min(1, kx))
        ky = max(0, min(1, ky))

        kp_strs.append(f"{kx:.6f} {ky:.6f} {kv}")

    # Build line: class_id cx cy w h kp0_x kp0_y kp0_v ...
    line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kp_strs)

    return line


def main():
    parser = argparse.ArgumentParser(description="DWPose Auto Labeling Tool - Crop Local Images")
    parser.add_argument("-i", "--images", required=True, help="Input images directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory (images/labels subdirs)")
    parser.add_argument("-d", "--detector", choices=["yolox", "yolov8", "yolo26"],
                        default="yolox", help="Detection model type")
    parser.add_argument("--padding", type=float, default=0.2, help="Crop box padding ratio")
    parser.add_argument("--save-vis", action="store_true", help="Save visualization images")
    args = parser.parse_args()

    input_dir = Path(args.images)
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if args.save_vis:
        vis_dir = output_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_exts:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    # Remove duplicates
    seen = set()
    unique_files = []
    for f in image_files:
        if f.name not in seen:
            seen.add(f.name)
            unique_files.append(f)
    image_files = sorted(unique_files)

    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images")
    print(f"Detector: {args.detector}")
    print(f"Padding: {args.padding}")
    print(f"Output images: {images_dir}")
    print(f"Output labels: {labels_dir}")
    print()

    # Load models
    det_session, pose_session, detector_type = load_models(args.detector)
    print()

    total_crops = 0

    # Process images
    for img_idx, img_path in enumerate(image_files):
        print(f"[{img_idx+1}/{len(image_files)}] Processing: {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Error: Cannot read image")
            continue

        # Run detection and pose
        result = process_image(img, det_session, pose_session, detector_type)

        if result is None or len(result[0]) == 0:
            print(f"  No persons detected")
            continue

        boxes, keypoints, scores = result
        print(f"  Detected {len(boxes)} persons")

        # Process each person
        for person_idx, box in enumerate(boxes):
            kps = keypoints[person_idx]  # [133, 2]
            scs = scores[person_idx]      # [133]

            # Crop and normalize
            crop_img, crop_box, local_kps, local_scs = crop_and_normalize_keypoints(
                img, box, kps, scs, padding=args.padding
            )

            if crop_img.shape[0] < 20 or crop_img.shape[1] < 20:
                print(f"    Person {person_idx+1}: crop too small, skipped")
                continue

            # Generate filename
            crop_name = f"{img_path.stem}_p{person_idx+1}"
            crop_img_path = images_dir / f"{crop_name}.jpg"
            label_path = labels_dir / f"{crop_name}.txt"

            # Save crop image
            cv2.imwrite(str(crop_img_path), crop_img)

            # Generate and save label
            line = convert_to_yolo_format_local(crop_img.shape, local_kps, local_scs)
            label_path.write_text(line)

            # Count visible keypoints
            visible_count = sum(1 for s in local_scs if s > 0.3)
            print(f"    Person {person_idx+1}: saved {crop_img.shape[1]}x{crop_img.shape[0]}, {visible_count} visible keypoints")

            total_crops += 1

            # Save visualization if requested
            if args.save_vis:
                vis_img = crop_img.copy()
                # Draw keypoints
                for j in range(len(local_kps)):
                    if local_scs[j] > 0.3:
                        x, y = int(local_kps[j, 0]), int(local_kps[j, 1])
                        cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)
                cv2.imwrite(str(vis_dir / f"{crop_name}.jpg"), vis_img)

    print()
    print("=" * 60)
    print("Done!")
    print(f"Total crops saved: {total_crops}")
    print(f"Images saved to: {images_dir}")
    print(f"Labels saved to: {labels_dir}")
    if args.save_vis:
        print(f"Visualizations saved to: {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
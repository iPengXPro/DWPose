#!/usr/bin/env python
"""
DWPose 推理脚本 - 叠加原图版本
将关键点骨架叠加在原图上显示
参照 ControlNet dwpose/util.py 的彩色绘制方式
"""

import os
import sys
import cv2
import numpy as np
import math
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "DWPose-models"
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "results_overlay"

OUTPUT_DIR.mkdir(exist_ok=True)

# 获取测试图片
test_images = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png"))

print("="*60)
print("DWPose 推理 - 叠加原图版")
print("="*60)
print(f"图片目录：{IMAGE_DIR}")
print(f"找到 {len(test_images)} 张图片")

# 加载模型
print("\n加载 ONNX 模型...")

try:
    import onnxruntime as ort
except ImportError:
    print("错误：需要安装 onnxruntime")
    sys.exit(1)

det_model = MODEL_DIR / "yolox_l.onnx"
pose_model = MODEL_DIR / "dw-ll_ucoco_384.onnx"

if not det_model.exists():
    print(f"错误：{det_model} 不存在")
    sys.exit(1)
if not pose_model.exists():
    print(f"错误：{pose_model} 不存在")
    sys.exit(1)

# 使用 CPU（避免 GPU 兼容性问题）
providers = ['CPUExecutionProvider']
print("使用设备：CPU")

session_det = ort.InferenceSession(str(det_model), providers=providers)
session_pose = ort.InferenceSession(str(pose_model), providers=providers)

# 导入推理函数
dwpose_path = BASE_DIR / "ControlNet-v1-1-nightly" / "annotator" / "dwpose"
sys.path.insert(0, str(dwpose_path))
from onnxdet import inference_detector
from onnxpose import inference_pose


# ========== 绘制函数 (参照 util.py) ==========

def draw_bodypose(canvas, keypoints, scores, score_thr=0.3):
    """
    绘制身体骨架
    keypoints: [N, 133, 2] - DWPose 输出
    scores: [N, 133]
    Body keypoints: 0-17
    """
    H, W = canvas.shape[:2]

    # DWPose body 关键点索引 (0-based):
    # 0:nose, 1:neck, 2:right_shoulder, 3:right_elbow, 4:right_wrist,
    # 5:left_shoulder, 6:left_elbow, 7:left_wrist, 8:right_hip, 9:right_knee,
    # 10:right_ankle, 11:left_hip, 12:left_knee, 13:left_ankle,
    # 14:right_eye, 15:left_eye, 16:right_ear, 17:left_ear

    # 骨架连接 (DWPose 0-based 索引，参照 util.py OpenPose limbSeq - 1)
    limbSeq = [
        [1, 2],   # neck(1) - right_shoulder(2)
        [1, 5],   # neck(1) - left_shoulder(5)
        [2, 3],   # right_shoulder(2) - right_elbow(3)
        [3, 4],   # right_elbow(3) - right_wrist(4)
        [5, 6],   # left_shoulder(5) - left_elbow(6)
        [6, 7],   # left_elbow(6) - left_wrist(7)
        [1, 8],   # neck(1) - right_hip(8)
        [8, 9],   # right_hip(8) - right_knee(9)
        [9, 10],  # right_knee(9) - right_ankle(10)
        [1, 11],  # neck(1) - left_hip(11)
        [11, 12], # left_hip(11) - left_knee(12)
        [12, 13], # left_knee(12) - left_ankle(13)
        [1, 0],   # neck(1) - nose(0)
        [0, 14],  # nose(0) - right_eye(14)
        [14, 16], # right_eye(14) - right_ear(16)
        [0, 15],  # nose(0) - left_eye(15)
        [15, 17], # left_eye(15) - left_ear(17)
        [2, 16],  # right_shoulder(2) - right_ear(16)
        [5, 17],  # left_shoulder(5) - left_ear(17)
    ]  # 19 limbs

    # 颜色 (BGR) - 19种颜色
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 255, 255]]

    stickwidth = 4

    for person_idx in range(len(keypoints)):
        kps = keypoints[person_idx]
        scs = scores[person_idx]

        # 绘制骨架线
        for i in range(len(limbSeq)):  # 绘制全部 19 个连接
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

            # 计算椭圆参数 (参照 util.py)
            # util.py: Y = candidate[:, 0] (列坐标), X = candidate[:, 1] (行坐标)
            # angle = atan2(X[0] - X[1], Y[0] - Y[1])
            Y = [x1, x2]  # 列坐标 (对应 util.py 的 Y)
            X = [y1, y2]  # 行坐标 (对应 util.py 的 X)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    # 混合背景 (参照 util.py: canvas = (canvas * 0.6).astype(np.uint8))
    canvas = (canvas * 0.6).astype(np.uint8)

    # 绘制关键点
    for person_idx in range(len(keypoints)):
        kps = keypoints[person_idx]
        scs = scores[person_idx]

        for i in range(18):  # 18 body keypoints
            if i >= len(kps):
                break
            if scs[i] < score_thr:
                continue
            x, y = int(kps[i][0]), int(kps[i][1])
            if x >= 0 and y >= 0:
                cv2.circle(canvas, (x, y), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, keypoints, scores, score_thr=0.3, is_left=True):
    """
    绘制手部骨架
    Left hand: 92-112 (21 points)
    Right hand: 113-132 (20 points)
    """
    H, W = canvas.shape[:2]

    # 手指连接
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for person_idx in range(len(keypoints)):
        kps = keypoints[person_idx]
        scs = scores[person_idx]

        # 获取手部关键点
        if is_left:
            hand_kps = kps[92:113]  # 21 points
            hand_scs = scs[92:113]
        else:
            hand_kps = kps[113:133]  # 20 points
            hand_scs = scs[113:133]

        # 绘制骨架线
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

            # HSV 颜色
            hue = ie / len(edges)
            # HSV to BGR
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))

            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), bgr, thickness=2)

    return canvas


def draw_facepose(canvas, keypoints, scores, score_thr=0.3):
    """
    绘制面部关键点
    Face: 24-91 (68 points)
    """
    H, W = canvas.shape[:2]

    for person_idx in range(len(keypoints)):
        kps = keypoints[person_idx]
        scs = scores[person_idx]

        face_kps = kps[24:92]  # 68 points
        face_scs = scs[24:92]

        for kpt, score in zip(face_kps, face_scs):
            if score < score_thr:
                continue
            x, y = int(kpt[0]), int(kpt[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)

    return canvas


def draw_pose_on_image(img, keypoints, scores, score_thr=0.3):
    """
    在原图上绘制完整姿态 (body + hands + face)
    keypoints: [N, 133, 2]
    scores: [N, 133]
    """
    canvas = img.copy()

    # 绘制身体
    canvas = draw_bodypose(canvas, keypoints, scores, score_thr)

    # 绘制左手
    canvas = draw_handpose(canvas, keypoints, scores, score_thr, is_left=True)

    # 绘制右手
    canvas = draw_handpose(canvas, keypoints, scores, score_thr, is_left=False)

    # 绘制面部
    canvas = draw_facepose(canvas, keypoints, scores, score_thr)

    return canvas


print(f"\n开始推理 {len(test_images)} 张图片...")

for img_path in test_images:
    print(f"\n处理：{img_path.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        print("  无法读取图片")
        continue

    H, W = img.shape[:2]

    # 检测
    dets = inference_detector(session_det, img)
    print(f"  检测到 {len(dets)} 个人")

    if len(dets) == 0:
        print("  未检测到人体")
        continue

    # 姿态
    keypoints, scores = inference_pose(session_pose, dets, img)

    # 关键点重映射：从 mmpose 顺序转换到 OpenPose 顺序 (参照 wholebody.py)
    keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)

    # 计算 neck 关键点 (从 left/right shoulder 平均)
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)  # mmpose: 5=left_shoulder, 6=right_shoulder
    neck[:, 2:4] = np.logical_and(
        keypoints_info[:, 5, 2:4] > 0.3,
        keypoints_info[:, 6, 2:4] > 0.3).astype(int)

    # 插入 neck 到索引 17
    new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

    # mmpose 到 OpenPose 的映射 (wholebody.py 第33-40行)
    # mmpose body 只有 17 个点，映射到 OpenPose 的 15 个点 + neck
    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]

    keypoints = new_keypoints_info[..., :2]
    scores = new_keypoints_info[..., 2]

    print(f"  关键点：{keypoints.shape}, 最高分：{scores.max():.3f}")

    # 绘制（叠加原图）
    result = draw_pose_on_image(img, keypoints, scores)

    # 保存
    out_path = OUTPUT_DIR / f"{img_path.stem}.jpg"
    cv2.imwrite(str(out_path), result)
    print(f"  保存：{out_path.name}")

print("\n" + "="*60)
print("完成!")
print(f"结果保存在：{OUTPUT_DIR}")
print("="*60)
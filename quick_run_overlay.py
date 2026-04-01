#!/usr/bin/env python
"""
DWPose 推理脚本 - 叠加原图版本
将关键点骨架叠加在原图上显示
"""

import os
import sys
import cv2
import numpy as np
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


def draw_pose_on_image(img, keypoints, scores, score_thr=0.3):
    """在原图上绘制关键点骨架"""
    H, W = img.shape[:2]
    canvas = img.copy()

    skeleton = [
        [0, 1], [0, 2], [1, 3], [2, 4],
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 11], [6, 12], [11, 12],
        [11, 13], [13, 15], [12, 14], [14, 16],
    ]

    # 颜色 (BGR)
    kp_color = (0, 255, 0)  # 绿色
    line_color = (0, 255, 0)

    for i in range(len(keypoints)):
        kps = keypoints[i]
        scs = scores[i]

        # 画点
        for j, (kpt, score) in enumerate(zip(kps, scs)):
            if score > score_thr:
                x, y = int(kpt[0]), int(kpt[1])
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(canvas, (x, y), 8, kp_color, -1)
                    cv2.circle(canvas, (x, y), 10, (255, 255, 255), 2)

        # 画线
        for e in skeleton:
            if e[0] < len(kps) and e[1] < len(kps):
                p1, p2 = kps[e[0]], kps[e[1]]
                s1, s2 = scs[e[0]], scs[e[1]]
                if s1 > score_thr and s2 > score_thr:
                    x1, y1 = int(p1[0]), int(p1[1])
                    x2, y2 = int(p2[0]), int(p2[1])
                    cv2.line(canvas, (x1, y1), (x2, y2), line_color, 3)

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
    kps, scores = inference_pose(session_pose, dets, img)
    print(f"  关键点：{kps.shape}, 最高分：{scores.max():.3f}")

    # 绘制（叠加原图）
    result = draw_pose_on_image(img, kps, scores)

    # 保存
    out_path = OUTPUT_DIR / f"{img_path.stem}.jpg"
    cv2.imwrite(str(out_path), result)
    print(f"  保存：{out_path.name}")

print("\n" + "="*60)
print("完成!")
print(f"结果保存在：{OUTPUT_DIR}")
print("="*60)

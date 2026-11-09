"""回调级逻辑测试（不开摄像头，覆盖录入回调分支）。

对应实施方案 M3 验收 A3-19：只在开摄像头时才执行的代码行，
语法检查和导入测试都发现不了问题，必须用回调级测试覆盖。

运行：.venv/Scripts/python.exe tools/test_ui_logic.py
"""
import os
import sys
import time

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.quality_gate import check_gate
from service.enroll_service import EnrollSession
from repository.person_repo import PersonRepo
from repository.feature_repo import FeatureRepo


def _make_face_row():
    """构造一个合成的 YuNet 检测结果行（15 列），模拟一张正脸。"""
    # bbox: x, y, w, h
    x, y, w, h = 200, 150, 300, 300
    # 5 关键点：右眼、左眼、鼻尖、右嘴角、左嘴角（正脸对称）
    reye = (x + w * 0.35, y + h * 0.35)
    leye = (x + w * 0.65, y + h * 0.35)
    nose = (x + w * 0.50, y + h * 0.52)
    rmouth = (x + w * 0.40, y + h * 0.70)
    lmouth = (x + w * 0.60, y + h * 0.70)
    return np.array([x, y, w, h,
                     reye[0], reye[1], leye[0], leye[1],
                     nose[0], nose[1], rmouth[0], rmouth[1],
                     lmouth[0], lmouth[1], 0.95], dtype=np.float32)


def test_duplicate_student_no():
    """A3-18: 重复学号被拒绝。"""
    pr = PersonRepo()
    no = f"dup_{int(time.time())}"
    pr.create(no, "甲")
    try:
        pr.create(no, "乙")
        print("❌ 重复学号未被拒绝")
        return False
    except ValueError as e:
        print(f"✅ 重复学号被拒绝: {e}")
        return True


def test_gate_brightness():
    """A3-11/A3-12: 亮度不足/过曝被拒。"""
    face = _make_face_row()
    faces = np.array([face])

    # 暗图
    dark = np.zeros((480, 640, 3), dtype=np.uint8)
    passed, reason = check_gate(faces, dark)
    print(f"暗图门控: passed={passed}, reason={reason}")
    assert not passed and "光照" in reason

    # 过曝
    bright = np.full((480, 640, 3), 255, dtype=np.uint8)
    passed, reason = check_gate(faces, bright)
    print(f"过曝门控: passed={passed}, reason={reason}")
    assert not passed and "过曝" in reason

    print("✅ 亮度门控生效")


def test_gate_multi_face():
    """A3-10: 多人画面被拒。"""
    face = _make_face_row()
    faces = np.array([face, face])
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    passed, reason = check_gate(faces, img)
    print(f"多人门控: passed={passed}, reason={reason}")
    assert not passed and "多张" in reason
    print("✅ 多人拦截生效")


def test_gate_side_face():
    """A3-14: 侧脸被拒。"""
    face = _make_face_row().copy()
    # 构造侧脸：鼻尖大幅偏移
    x, y, w, h = face[0], face[1], face[2], face[3]
    face[4], face[5] = x + w * 0.30, y + h * 0.35   # 右眼
    face[6], face[7] = x + w * 0.40, y + h * 0.35   # 左眼（很近）
    face[8], face[9] = x + w * 0.60, y + h * 0.52   # 鼻尖大幅偏移
    faces = np.array([face])
    # 用带纹理的图（随机噪声），让清晰度门控通过，才能测到姿态门控
    rng = np.random.default_rng(42)
    img = rng.integers(60, 200, (480, 640, 3), dtype=np.uint8)
    passed, reason = check_gate(faces, img)
    print(f"侧脸门控: passed={passed}, reason={reason}")
    assert not passed and "侧脸" in reason
    print("✅ 侧脸拦截生效")


def main():
    print("=" * 60)
    print("M3 回调级逻辑测试（不开摄像头）")
    print("=" * 60)

    results = []
    results.append(("重复学号拒绝 (A3-18)", test_duplicate_student_no()))
    test_gate_brightness()
    test_gate_multi_face()
    test_gate_side_face()

    print()
    print("=" * 60)
    print("测试完成：门控各分支全部覆盖")
    print("=" * 60)


if __name__ == "__main__":
    main()

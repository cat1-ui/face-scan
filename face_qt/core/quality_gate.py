"""质量门控（G0~G5）。

核心职责（实施方案 M3 T3-2）：
- 检查顺序：G0 人数 → G1 亮度（前置）→ G2 清晰度 → G3 姿态 → G4 脸太小 → G5 脸太大
- 每条拒因都是「全中文可执行」的原因

关键点：
- 亮度必须排在清晰度之前：低光是清晰度低的上游原因。
- 所有阈值来自 core.config（清晰度阈值已由 M1 实测定档）。
"""
import cv2
import numpy as np

import face_metrics as fm
from core import config


def check_gate(faces, img_bgr):
    """执行 G0~G5 门控。

    参数：
        faces: YuNet 检测结果（N×15 数组或 None）
        img_bgr: 当前帧（BGR）

    返回：
        (passed: bool, reason: str)
        passed=True 表示通过，reason=""；
        passed=False 表示拒绝，reason 为可执行中文原因。
    """
    # G0 人数检查
    if faces is None or len(faces) == 0:
        return False, "未检测到人脸，请正对摄像头"
    if len(faces) >= 2:
        return False, "画面中有多张人脸，请保证只有一人"

    face = faces[0]
    bbox = face[0:4]
    score = float(face[14])

    # G1 亮度检查（前置）
    m = fm.compute_all(img_bgr, bbox=bbox)
    brightness = m["brightness"]
    if brightness < config.BRIGHTNESS_MIN:
        return False, "光照不足，请开灯或转向光源"
    if brightness > config.BRIGHTNESS_MAX:
        return False, "画面过曝，请背对光源"

    # G2 清晰度检查
    sharpness = m["sharpness"]
    if sharpness < config.SHARPNESS_MIN:
        return False, "画面模糊，请稳住并擦净镜头"

    # G3 姿态检查（侧脸）：基于关键点对称性估算
    if _is_side_face(face):
        return False, "检测到侧脸，请正对摄像头"

    # G4 脸太小
    face_w = m["face_width"]
    if face_w is not None and face_w < config.FACE_WIDTH_MIN:
        return False, "脸太小，请靠近摄像头"

    # G5 脸太大（超过画面宽度的一半）
    if face_w is not None and img_bgr.shape[1] > 0 and \
            face_w > img_bgr.shape[1] * 0.6:
        return False, "脸太大，请稍微后退"

    return True, ""


def _is_side_face(face):
    """基于 5 关键点估算是否侧脸/歪头。

    判据（考核要求，由 5 关键点推导）：
    - 两眼连线倾角 > EYE_TILT_MAX（20°）→ 头歪
    - 鼻尖横向偏移/脸宽 > NOSE_OFFSET_MAX（0.2）→ 侧脸
    """
    tilt = fm.eye_tilt_angle(face)
    if tilt is not None and tilt > config.EYE_TILT_MAX:
        return True

    offset = fm.nose_offset_ratio(face)
    if offset is not None and offset > config.NOSE_OFFSET_MAX:
        return True

    return False

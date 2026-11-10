"""图像质量指标计算（纯函数，无副作用）。

四个指标：清晰度 / 亮度 / 对比度 / 色温。

关键点：
- 清晰度必须「归一化」——先把人脸 ROI 缩放到 112×112 再算拉普拉斯，
  否则固定阈值对大脸过严、对小脸过松（设计方案 3.5 节）。
- 亮度检查应「前置」——低光是清晰度低的上游原因。
"""
import cv2
import numpy as np


def _to_gray(img_bgr):
    """转灰度。入参约定为 BGR（与 OpenCV 一致）。"""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def sharpness_laplacian(img_bgr, normalize_size=112):
    """清晰度：拉普拉斯方差。

    若提供 normalize_size，则先把图像缩放到该尺寸再算，实现归一化。
    加 3×3 高斯平滑抑制传感器噪声对高频的干扰（对齐老师实现）。
    值越大越清晰。
    """
    gray = _to_gray(img_bgr)
    if normalize_size is not None:
        gray = cv2.resize(gray, (normalize_size, normalize_size),
                          interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(img_bgr):
    """亮度：灰度均值（0~255）。"""
    return float(_to_gray(img_bgr).mean())


def contrast(img_bgr):
    """对比度：灰度标准差。"""
    return float(_to_gray(img_bgr).std())


def color_temperature(img_bgr):
    """色温：红蓝通道均值比。

    返回 r/b 比值。>1 偏暖（偏黄），<1 偏冷（偏蓝）。仅供参考。
    """
    b, g, r = cv2.split(img_bgr)
    r_mean = float(r.mean())
    b_mean = float(b.mean())
    # 避免除零
    if b_mean < 1e-6:
        return 1.0
    return r_mean / b_mean


def face_width(bbox):
    """人脸框宽度（像素）。bbox 为 (x, y, w, h)。"""
    return float(bbox[2])


# ---------------- 姿态（由 YuNet 5 关键点推导）----------------
# 关键点索引（face 数组，每点占 (x, y) 两个元素）：
#   右眼=0  左眼=1  鼻尖=2  右嘴角=3  左嘴角=4
def _landmarks(face):
    """从 N×15 的 face 行提取 5 关键点坐标列表 [(x,y)*5]。"""
    pts = []
    for i in range(5):
        pts.append((float(face[4 + 2 * i]), float(face[5 + 2 * i])))
    return pts


def eye_tilt_angle(face):
    """两眼连线倾角（度，取绝对值 0~90）。

    由右眼→左眼连线与水平方向的夹角计算，反映头部左右倾斜。
    关键点顺序固定：右眼(0)、左眼(1)。
    返回 None 表示关键点无效（两眼重合）。
    """
    pts = _landmarks(face)
    reye, leye = pts[0], pts[1]
    dx = leye[0] - reye[0]
    dy = leye[1] - reye[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 1e-6:
        return None
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    # 归一化到 0~90（atan2 可能给 >90 的钝角，取补角）
    if angle > 90:
        angle = 180 - angle
    return float(angle)


def nose_offset_ratio(face):
    """鼻尖横向偏移 / 脸宽 的比值（0~∞）。

    用「鼻尖横坐标」相对「两眼连线中点横坐标」的偏移量，
    除以脸宽（此处以两眼间距作为脸宽参考，因框宽含发际/下巴噪声）。
    偏移越大说明脸转得越侧。返回 None 表示无效。
    """
    pts = _landmarks(face)
    reye, leye, nose = pts[0], pts[1], pts[2]
    face_w = abs(leye[0] - reye[0])   # 两眼间距近似脸宽
    if face_w < 1e-6:
        return None
    eye_mid_x = (reye[0] + leye[0]) / 2.0
    return float(abs(nose[0] - eye_mid_x) / face_w)


def compute_all(img_bgr, bbox=None):
    """一次性计算所有指标。

    bbox 为 (x, y, w, h)。若提供，则清晰度/亮度/对比度/色温都在人脸 ROI 上算，
    脸宽从 bbox 取；否则全图计算，脸宽为 None。
    """
    roi = img_bgr
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_bgr.shape[1], x + w)
        y2 = min(img_bgr.shape[0], y + h)
        if x2 <= x or y2 <= y:
            # 非法框，退回全图
            roi = img_bgr
            fw = None
        else:
            roi = img_bgr[y:y2, x:x2]
            fw = float(x2 - x)
    else:
        fw = None

    return {
        "sharpness": sharpness_laplacian(roi),
        "brightness": brightness(roi),
        "contrast": contrast(roi),
        "color_temp": color_temperature(roi),
        "face_width": fw,
    }

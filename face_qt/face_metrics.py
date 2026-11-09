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

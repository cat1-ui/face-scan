"""人脸特征提取封装（SFace / cv2.FaceRecognizerSF）。

核心职责（实施方案 M3 T3-1）：
- alignCrop：检测结果 → 112×112 对齐正脸
- feature：对齐图 → 128 维特征向量
- 出口统一 L2 归一化（硬性门槛 A3-2）

关键点：
- alignCrop 的 face_box 要传「检测结果的整行」（15 列），不只是 bbox。
- feature 输出 128 维，必须在出口做 L2 归一化，余弦相似度才有意义。
"""
import cv2
import numpy as np

from core import config


class FaceRecognizer:
    """SFace 特征提取器封装。"""

    def __init__(self, model_path=None):
        model_path = model_path or f"{config.MODELS_DIR}/{config.REC_MODEL}"
        self._rec = cv2.FaceRecognizerSF.create(model_path, "")

    def align_crop(self, img_bgr, face_row):
        """对齐裁剪。face_row 为 YuNet 检测结果的单行（15 列）。

        返回 112×112 对齐图。
        """
        aligned = self._rec.alignCrop(img_bgr, face_row)
        return aligned

    def feature(self, aligned_img):
        """从对齐图提取 128 维特征，并做 L2 归一化。

        返回 shape=(128,) 的 float32 数组，范数为 1。
        """
        feat = self._rec.feature(aligned_img)  # (1, 128) 或 (128,)
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        # L2 归一化
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat.astype(np.float32)

    def align_and_feature(self, img_bgr, face_row):
        """一步完成：对齐 + 特征 + 归一化。"""
        aligned = self.align_crop(img_bgr, face_row)
        feat = self.feature(aligned)
        return aligned, feat

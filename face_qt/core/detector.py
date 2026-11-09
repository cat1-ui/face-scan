"""人脸检测封装（YuNet / cv2.FaceDetectorYN）。

核心职责（实施方案 M2 T2-1 ~ T2-4）：
- 封装 FaceDetector 类：加载模型、setInputSize、detect
- draw()：画框 + 5 关键点 + 置信度
- draw_face_size_bar()：人脸尺寸实时提示条
- 统一通道处理

关键坑（设计方案第四章）：
1. setInputSize 每帧必调，传当前图像真实 (w, h)。
2. detect() 可能返回 None（不是空数组），必须判断。
3. 通道顺序：OpenCV 用 BGR，入口统一归一。
4. cv2.putText / rectangle 是原地修改并返回 None，不能写成赋值。
"""
import cv2
import numpy as np

from core import config

# 5 关键点固定顺序：右眼、左眼、鼻尖、右嘴角、左嘴角
KEYPOINT_NAMES = ["右眼", "左眼", "鼻尖", "右嘴角", "左嘴角"]


class FaceDetector:
    """YuNet 人脸检测器封装。"""

    def __init__(self, model_path=None, score_threshold=None):
        model_path = model_path or f"{config.MODELS_DIR}/{config.DET_MODEL}"
        self.score_threshold = score_threshold or config.DET_SCORE_THRESHOLD
        # 用临时尺寸创建，实际每帧 setInputSize 更新
        self._det = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), self.score_threshold)

    def set_input_size(self, w, h):
        self._det.setInputSize((int(w), int(h)))

    def detect(self, img_bgr):
        """检测人脸。返回 (faces, img) 中 faces 为 N×15 数组或 None。"""
        h, w = img_bgr.shape[:2]
        self.set_input_size(w, h)
        _, faces = self._det.detect(img_bgr)
        return faces

    @staticmethod
    def largest(faces):
        """取面积最大的那张脸（多人入镜时以它为主目标）。"""
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    # ---------------- 绘制 ----------------
    def draw(self, canvas, faces):
        """在 canvas（BGR）上画框 + 5 关键点 + 置信度。原地修改，返回 canvas。

        注意：不改变 canvas 引用，避免 putText 原地修改返回 None 的坑。
        """
        if faces is None or len(faces) == 0:
            cv2.putText(canvas, "未检测到人脸", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return canvas

        for face in faces:
            x, y, w, h = face[0:4].astype(int)
            score = float(face[14])

            # 画框（绿色）
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 置信度
            cv2.putText(canvas, f"{score:.2f}", (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 5 关键点
            for i in range(5):
                px, py = face[4 + 2 * i], face[5 + 2 * i]
                cv2.circle(canvas, (int(px), int(py)), 3, (0, 0, 255), -1)

        return canvas

    def draw_face_size_bar(self, canvas, face_width, min_width=None,
                           max_width=None):
        """在 canvas 顶部画人脸尺寸提示条。

        face_width 为当前人脸框宽度（像素）。低于 min_width 提示靠近。
        """
        min_width = min_width or config.FACE_WIDTH_MIN
        max_width = max_width or (min_width * 3)

        h = canvas.shape[0]
        bar_w = canvas.shape[1] - 40
        bar_h = 14
        bar_x, bar_y = 20, 20

        # 背景条
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (200, 200, 200), -1)

        # 有效区间（绿色）
        ratio = min(max(face_width / max_width, 0.0), 1.0)
        green_w = int(bar_w * ratio)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + green_w, bar_y + bar_h),
                      (0, 200, 0), -1)

        # 阈值刻度线
        thresh_x = bar_x + int(bar_w * (min_width / max_width))
        cv2.line(canvas, (thresh_x, bar_y - 3), (thresh_x, bar_y + bar_h + 3),
                 (0, 0, 255), 2)

        # 文字
        if face_width < min_width:
            tip = f"脸太小 ({face_width:.0f}px)，请靠近"
            color = (0, 0, 255)
        else:
            tip = f"人脸尺寸 {face_width:.0f}px"
            color = (0, 180, 0)
        cv2.putText(canvas, tip, (bar_x, bar_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return canvas

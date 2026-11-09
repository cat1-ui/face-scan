"""人脸检测页（M2）。

实时画面 + 检测框/关键点 + 置信度 + 人脸尺寸条 + FPS + 图片上传检测。
核心逻辑复用 core/detector.py，UI 只做展示与交互。
"""
import os
import time

import cv2
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QFrame, QCheckBox
)

from core import config
from core.detector import FaceDetector
from camera_device import CameraDevice


class DetectWorker(QThread):
    """后台线程：读帧 + 检测 + 绘制。发回标注后的帧。"""
    frame_ready = pyqtSignal(object, float, int)   # (canvas_bgr, fps, face_count)

    def __init__(self, device, detector, parent=None):
        super().__init__(parent)
        self.device = device
        self.detector = detector
        self._running = False
        self._fps = 0.0
        self._last = time.time()
        self._frames = 0

    def run(self):
        self._running = True
        while self._running:
            ok, frame = self.device.read_frame()
            if not ok:
                continue
            # 检测 + 绘制
            faces = self.detector.detect(frame)
            self.detector.draw(frame, faces)

            face_count = 0 if faces is None else len(faces)
            if face_count >= 1:
                self.detector.draw_face_size_bar(frame, faces[0][2])

            # FPS 统计（每秒更新一次）
            self._frames += 1
            now = time.time()
            if now - self._last >= 1.0:
                self._fps = self._frames / (now - self._last)
                self._frames = 0
                self._last = now

            self.frame_ready.emit(frame, self._fps, face_count)

    def stop(self):
        self._running = False


class DetectPage(QWidget):
    def __init__(self):
        super().__init__()
        self.device = None
        self.worker = None
        self.detector = FaceDetector()
        self._last_canvas = None
        self._build_ui()

    def _build_ui(self):
        # 按钮行
        self.btn_start = QPushButton("打开摄像头")
        self.btn_stop = QPushButton("关闭摄像头")
        self.btn_stop.setEnabled(False)
        self.btn_upload = QPushButton("图片上传检测")
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_upload.clicked.connect(self.on_upload)

        self.chk_size_bar = QCheckBox("显示尺寸条")
        self.chk_size_bar.setChecked(True)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_upload)
        btn_row.addWidget(self.chk_size_bar)
        btn_row.addStretch()

        # 画面
        self.img_label = QLabel("点击「打开摄像头」开始实时检测")
        self.img_label.setFrameShape(QFrame.Shape.Box)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setMinimumSize(720, 420)

        # 状态栏
        self.status_label = QLabel("状态: 未启动 | FPS: -- | 人脸数: --")
        self.status_label.setStyleSheet("color: #555;")

        layout = QVBoxLayout()
        layout.addLayout(btn_row)
        layout.addWidget(self.img_label, 5)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._refresh)
        self.ui_timer.setInterval(30)

    # ---------------- 摄像头 ----------------
    def on_start(self):
        self.btn_start.setEnabled(False)
        # 探测 + 正式打开（复用 M1 验证过的顺序：探测完成后再正式打开）
        probe = CameraDevice(0)
        if not probe.open():
            self.status_label.setText("状态: ❌ 摄像头打开失败（被占用或不存在）")
            self.btn_start.setEnabled(True)
            return
        recs = probe.probe_resolutions()
        hw = probe.hardware_max_resolution(recs)
        probe.release()

        self.device = CameraDevice(0)
        if hw:
            self.device.open(width=hw[0], height=hw[1])
        else:
            self.device.open()
        if self.device.cap is None:
            self.status_label.setText("状态: ❌ 打开失败")
            self.btn_start.setEnabled(True)
            return
        for _ in range(5):   # 预热
            self.device.read_frame()

        self.worker = DetectWorker(self.device, self.detector)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.start()
        self.ui_timer.start()
        self.btn_stop.setEnabled(True)
        self.status_label.setText("状态: 检测中...")

    def _on_frame(self, canvas, fps, face_count):
        self._last_canvas = canvas
        self.status_label.setText(
            f"状态: 检测中 | FPS: {fps:.1f} | 人脸数: {face_count}")

    def _refresh(self):
        if self._last_canvas is None:
            return
        frame = self._last_canvas
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.img_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def on_stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self.ui_timer.stop()
        if self.device is not None:
            self.device.release()
            self.device = None
        self._last_canvas = None
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.status_label.setText("状态: 已停止（设备已释放）")

    # ---------------- 图片上传 ----------------
    def on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.status_label.setText("状态: ❌ 图片读取失败")
            return

        # 异常输入防护（A2-9）
        try:
            if img.size == 0 or img.shape[0] < 20 or img.shape[1] < 20:
                self.status_label.setText("状态: ❌ 图片尺寸过小")
                return
            faces = self.detector.detect(img)
            canvas = self.detector.draw(img.copy(), faces)
            n = 0 if faces is None else len(faces)
            self.status_label.setText(f"状态: 图片检测完成 | 人脸数: {n}")
            # 展示结果
            h, w = canvas.shape[:2]
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.img_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        except cv2.error as e:
            self.status_label.setText(f"状态: ❌ 检测异常: {e}")

    def hideEvent(self, event):
        """切走页面时释放摄像头。"""
        self.on_stop()
        super().hideEvent(event)

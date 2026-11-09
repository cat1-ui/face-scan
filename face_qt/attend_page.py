"""人脸识别页（M4）。

实时画面 + 识别结果（姓名 + 相似度）+ 打卡提示 + 拒识/相似脸提示。
"""
import time

import cv2
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
)

import face_metrics as fm
from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.quality_gate import check_gate
from core.matcher import Matcher
from service.attend_service import AttendService
from camera_device import CameraDevice


class AttendWorker(QThread):
    """后台线程：读帧 + 检测 + 门控 + 识别。发回状态。"""
    frame_ready = pyqtSignal(object, dict)   # (canvas_bgr, result)

    def __init__(self, device, detector, recognizer, attend_service, parent=None):
        super().__init__(parent)
        self.device = device
        self.detector = detector
        self.recognizer = recognizer
        self.attend_service = attend_service
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            ok, frame = self.device.read_frame()
            if not ok:
                continue

            faces = self.detector.detect(frame)
            result = None

            if faces is None or len(faces) == 0:
                result = {"action": "noface"}
            else:
                passed, reason = check_gate(faces, frame)
                if not passed:
                    result = {"action": "badq", "reason": reason}
                else:
                    face = faces[0]
                    aligned, feat = self.recognizer.align_and_feature(
                        frame, face)
                    m = fm.compute_all(frame, bbox=face[0:4])
                    result = self.attend_service.process(
                        feat, sharpness=m["sharpness"],
                        brightness=m["brightness"])

            # 绘制
            self.detector.draw(frame, faces)
            if faces is not None and len(faces) >= 1:
                self.detector.draw_face_size_bar(frame, faces[0][2])

            self.frame_ready.emit(frame, result)

    def stop(self):
        self._running = False


class AttendPage(QWidget):
    def __init__(self):
        super().__init__()
        self.device = None
        self.worker = None
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.matcher = Matcher()
        self.attend_service = AttendService(self.matcher)
        self._last_canvas = None
        self._build_ui()

    def _build_ui(self):
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始识别")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()

        self.result_label = QLabel("识别结果：--")
        self.result_label.setStyleSheet(
            "font-size: 18px; color: #333; padding: 8px;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.img_label = QLabel("识别画面")
        self.img_label.setFrameShape(QFrame.Shape.Box)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setMinimumSize(640, 420)

        layout = QVBoxLayout()
        layout.addLayout(btn_row)
        layout.addWidget(self.result_label)
        layout.addWidget(self.img_label, 5)
        self.setLayout(layout)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._refresh)
        self.ui_timer.setInterval(30)

    def on_start(self):
        # 刷新 gallery（可能刚录入了新人）
        self.matcher.load_gallery()
        if self.matcher.gallery.shape[0] == 0:
            self.result_label.setText(
                "识别结果：库里暂无人员，请先到「人脸录入」页录入")
            return

        probe = CameraDevice(0)
        if not probe.open():
            self.result_label.setText("识别结果：摄像头打开失败")
            return
        recs = probe.probe_resolutions()
        hw = probe.hardware_max_resolution(recs)
        probe.release()

        self.device = CameraDevice(0)
        self.device.open(width=hw[0], height=hw[1]) if hw else self.device.open()
        if self.device.cap is None:
            self.result_label.setText("识别结果：打开失败")
            return
        for _ in range(5):
            self.device.read_frame()

        self.worker = AttendWorker(
            self.device, self.detector, self.recognizer, self.attend_service)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.start()
        self.ui_timer.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _on_frame(self, canvas, result):
        self._last_canvas = canvas
        if result is None:
            return
        action = result.get("action")
        if action == "noface":
            self.result_label.setText("识别结果：请进入画面")
        elif action == "badq":
            self.result_label.setText(f"识别结果：{result.get('reason')}")
        elif action == "unknown":
            sim = result.get("top1_sim", 0)
            self.result_label.setText(
                f"识别结果：未识别（相似度 {sim:.2f}）")
        elif action == "confirm":
            self.result_label.setText(
                f"识别结果：疑似 {result.get('name')}，请人工确认"
                f"（top1={result.get('top1_sim',0):.2f} "
                f"top2={result.get('top2_sim',0):.2f}）")
        elif action == "already":
            self.result_label.setText(
                f"识别结果：{result.get('name')}（本时段已记录）")
        elif action == "recorded":
            self.result_label.setText(
                f"识别结果：✅ {result.get('name')} 打卡成功"
                f"（相似度 {result.get('top1_sim',0):.2f}）")

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

    def hideEvent(self, event):
        self.on_stop()
        super().hideEvent(event)

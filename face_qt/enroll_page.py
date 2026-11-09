"""人脸录入页（M3）。

表单（学号/姓名/院系）+ 实时预览 + 指令提示 + 采样计数 + 诊断面板。
采集流程由 EnrollSession 状态机驱动，指令自动切换 + 倒计时提示。
"""
import time

import cv2
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFrame, QTextEdit
)

from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.quality_gate import check_gate
from service.enroll_service import EnrollSession
from camera_device import CameraDevice


class EnrollWorker(QThread):
    """后台线程：读帧 + 检测 + 门控 + 录入。发回状态。"""
    frame_ready = pyqtSignal(object, dict)   # (canvas_bgr, enroll_status)

    def __init__(self, device, detector, session, parent=None):
        super().__init__(parent)
        self.device = device
        self.detector = detector
        self.session = session
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            ok, frame = self.device.read_frame()
            if not ok:
                continue

            faces = self.detector.detect(frame)

            # 门控 + 录入（只在该指令未完成时处理）
            if not self.session.finished:
                passed, reason = check_gate(faces, frame)
                if passed:
                    # 通过门控才提取特征 + 入库
                    status = self.session.process_frame(frame, faces)
                else:
                    status = self.session._status(reason)
            else:
                status = self.session._status("样本已采集完成（5/5）")

            # 绘制：画框 + 关键点
            self.detector.draw(frame, faces)
            if faces is not None and len(faces) >= 1:
                self.detector.draw_face_size_bar(frame, faces[0][2])

            self.frame_ready.emit(frame, status)

    def stop(self):
        self._running = False


class EnrollPage(QWidget):
    def __init__(self):
        super().__init__()
        self.device = None
        self.worker = None
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.session = None
        self._last_canvas = None
        self._build_ui()

    def _build_ui(self):
        # 表单行
        self.ed_student_no = QLineEdit()
        self.ed_student_no.setPlaceholderText("学号")
        self.ed_name = QLineEdit()
        self.ed_name.setPlaceholderText("姓名")
        self.ed_dept = QLineEdit()
        self.ed_dept.setPlaceholderText("院系（可选）")

        self.btn_start = QPushButton("开始录入")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)

        form_row = QHBoxLayout()
        form_row.addWidget(QLabel("学号:"))
        form_row.addWidget(self.ed_student_no)
        form_row.addWidget(QLabel("姓名:"))
        form_row.addWidget(self.ed_name)
        form_row.addWidget(QLabel("院系:"))
        form_row.addWidget(self.ed_dept)
        form_row.addWidget(self.btn_start)
        form_row.addWidget(self.btn_stop)

        # 指令提示
        self.inst_label = QLabel("请先填写学号姓名，然后点击「开始录入」")
        self.inst_label.setStyleSheet(
            "font-size: 16px; color: #333; padding: 6px;")

        # 画面
        self.img_label = QLabel("录入画面")
        self.img_label.setFrameShape(QFrame.Shape.Box)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setMinimumSize(640, 400)

        # 采样计数
        self.count_label = QLabel("已保存: 0/5")

        # 诊断面板
        self.diag_text = QTextEdit()
        self.diag_text.setReadOnly(True)
        self.diag_text.setMaximumHeight(120)
        self.diag_text.setPlaceholderText("诊断信息将在此显示")

        layout = QVBoxLayout()
        layout.addLayout(form_row)
        layout.addWidget(self.inst_label)
        layout.addWidget(self.img_label, 4)
        layout.addWidget(self.count_label)
        layout.addWidget(self.diag_text)
        self.setLayout(layout)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._refresh)
        self.ui_timer.setInterval(30)

    def on_start(self):
        student_no = self.ed_student_no.text().strip()
        name = self.ed_name.text().strip()
        dept = self.ed_dept.text().strip() or None

        if not student_no or not name:
            self.diag_text.setPlainText("❌ 请填写学号和姓名")
            return

        # 创建录入会话（重复学号会抛异常）
        try:
            self.session = EnrollSession(
                student_no, name, dept,
                recognizer=self.recognizer, detector=self.detector)
        except ValueError as e:
            self.diag_text.setPlainText(f"❌ {e}")
            return

        # 打开摄像头
        probe = CameraDevice(0)
        if not probe.open():
            self.diag_text.setPlainText("❌ 摄像头打开失败")
            return
        recs = probe.probe_resolutions()
        hw = probe.hardware_max_resolution(recs)
        probe.release()

        self.device = CameraDevice(0)
        self.device.open(width=hw[0], height=hw[1]) if hw else self.device.open()
        if self.device.cap is None:
            self.diag_text.setPlainText("❌ 打开失败")
            return
        for _ in range(5):
            self.device.read_frame()

        self.worker = EnrollWorker(self.device, self.detector, self.session)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.start()
        self.ui_timer.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.inst_label.setText(self.session.current_instruction_text())
        self.diag_text.setPlainText(f"开始录入 {student_no} {name}")

    def _on_frame(self, canvas, status):
        self._last_canvas = canvas
        # 更新指令提示
        if status.get("finished"):
            self.inst_label.setText("✅ " + status.get("message", "完成"))
        else:
            self.inst_label.setText(status.get("instruction", ""))
        self.count_label.setText(
            f"已保存: {status.get('saved_total', 0)}/5")
        # 诊断面板只显示最新消息（保留历史）
        msg = status.get("message", "")
        if msg:
            cur = self.diag_text.toPlainText()
            # 避免刷屏，只保留最近几条
            lines = cur.split("\n")[-6:]
            lines.append(f"[{status.get('instruction_idx', 0)+1}] {msg}")
            self.diag_text.setPlainText("\n".join(lines))

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

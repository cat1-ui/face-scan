"""设备体检 PyQt6 界面。

两个按钮（开始检测 / 关闭摄像头）+ 实时画面 + 实时指标 + 报告区。
核心逻辑复用 camera_device / face_metrics / report，UI 只做展示。

运行：.venv/Scripts/python.exe device_check_ui.py

对应 M1 验收：A1-1（实时画面）、A1-10（关闭真释放）、A1-12（未开始就关闭不崩溃）。
"""
import sys
import time

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QTextEdit, QFrame
)

import face_metrics as fm
import report as rp
from camera_device import CameraDevice
from core import config


class CameraWorker(QThread):
    """后台线程：循环读帧 + 算指标，通过信号发回主线程。"""
    frame_ready = pyqtSignal(object, dict)   # (frame_bgr, metrics_dict)
    finished_sampling = pyqtSignal()

    def __init__(self, device, parent=None):
        super().__init__(parent)
        self.device = device
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            ok, frame = self.device.read_frame()
            if not ok:
                continue
            # 人脸检测 + 指标
            h, w = frame.shape[:2]
            m = None
            # 简化：不在这线程加载检测模型，只算全图指标 + 简单人脸框
            # 检测模型在主流程采样时统一做；这里实时展示只算亮度/清晰度
            m = fm.compute_all(frame, bbox=None)
            self.frame_ready.emit(frame, m)

    def stop(self):
        self._running = False


class DeviceCheckWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("设备体检 · 人脸识别系统")
        self.resize(900, 640)

        self.device = None
        self.worker = None
        self.samples = []
        self.total_frames = 0
        self.valid_frames = 0
        self.start_time = None

        self._build_ui()

    def _build_ui(self):
        # 顶部按钮
        self.btn_start = QPushButton("开始检测")
        self.btn_stop = QPushButton("关闭摄像头")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()

        # 实时画面
        self.img_label = QLabel("点击「开始检测」启动摄像头")
        self.img_label.setFrameShape(QFrame.Shape.Box)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setMinimumSize(640, 360)

        # 实时指标
        self.metric_label = QLabel("清晰度: --\n亮度: --\n对比度: --\n色温: --")
        self.metric_label.setFrameShape(QFrame.Shape.Box)
        self.metric_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 报告区
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText("检测结束后在此显示体检报告")

        # 布局
        mid_row = QHBoxLayout()
        mid_row.addWidget(self.img_label, 3)
        mid_row.addWidget(self.metric_label, 1)

        layout = QVBoxLayout()
        layout.addLayout(btn_row)
        layout.addLayout(mid_row, 4)
        layout.addWidget(QLabel("体检报告："))
        layout.addWidget(self.report_text, 3)
        self.setLayout(layout)

        # 定时刷新画面
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._refresh_frame)
        self.ui_timer.setInterval(30)

    # ---------------- 回调 ----------------
    def on_start(self):
        """开始检测：探测设备 → 打开采样句柄 → 采样。"""
        self.btn_start.setEnabled(False)
        self.report_text.clear()
        self.samples = []
        self.total_frames = 0
        self.valid_frames = 0

        # 先用临时句柄探测分辨率与编码（探测阶段不持有正式句柄）
        probe = CameraDevice(index=0)
        if not probe.open():
            self.report_text.setPlainText(
                "❌ 摄像头打开失败：可能被占用或不存在。\n"
                "排查：关闭微信/钉钉/Zoom；检查设备管理器；换 index。")
            self.btn_start.setEnabled(True)
            return
        self.res_records = probe.probe_resolutions()
        hw = probe.hardware_max_resolution(self.res_records)
        self.codec_records = probe.probe_codecs(
            hw[0], hw[1]) if hw else []
        probe.release()

        # 所有探测完成后，正式打开采样句柄
        self.device = CameraDevice(index=0)
        if hw:
            self.device.open(width=hw[0], height=hw[1])
        else:
            self.device.open()
        if self.device.cap is None:
            self.report_text.setPlainText("❌ 以目标分辨率打开摄像头失败。")
            self.btn_start.setEnabled(True)
            return

        # 预热：读几帧让摄像头稳定
        for _ in range(5):
            self.device.read_frame()

        self.start_time = time.time()
        self.worker = CameraWorker(self.device)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.start()

        self.ui_timer.start()
        self.btn_stop.setEnabled(True)

    def _on_frame(self, frame, metrics):
        self.total_frames += 1
        self._last_frame = frame
        self._last_metrics = metrics
        # 有指标就累积（简化：全图指标也计入样本）
        if metrics and metrics.get("sharpness") is not None:
            self.valid_frames += 1
            self.samples.append(metrics)

    def _refresh_frame(self):
        if not hasattr(self, "_last_frame"):
            return
        frame = self._last_frame
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.img_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

        m = self._last_metrics
        self.metric_label.setText(
            f"清晰度: {m['sharpness']:.1f}\n"
            f"亮度: {m['brightness']:.1f}\n"
            f"对比度: {m['contrast']:.1f}\n"
            f"色温: {m['color_temp']:.2f}\n"
            f"采样帧: {self.total_frames}")

    def on_stop(self):
        """关闭摄像头：真正释放设备。"""
        # 停止 worker
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

        self.ui_timer.stop()

        if self.device is not None:
            duration = time.time() - self.start_time if self.start_time else 0
            fps = self.total_frames / duration if duration > 0 else 0
            self.device.fps = round(fps, 1)
            self.device.release()
            self.device = None

        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)

        # 生成报告
        if self.valid_frames > 0:
            report = rp.generate_report(
                self._build_report_dev(),
                self.samples, self.res_records, self.codec_records,
                self.total_frames, self.valid_frames,
                self.valid_frames / self.total_frames if self.total_frames else 0,
                duration)
            txt = rp.render_text(report)
            self.report_text.setPlainText(txt)
            path = rp.export_report(report)
            self.report_text.append(f"\n报告已导出: {path}")
        else:
            self.report_text.setPlainText("没有采到有效帧，无法生成报告。")

    def _build_report_dev(self):
        """构造报告所需的设备信息对象（基于本次探测结果）。"""
        # 硬件上限取探测出的最高 OK 档
        hw = self.res_records
        w = h = 0
        for r in hw:
            if r["ok"]:
                w, h = r["width_actual"], r["height_actual"]
                break
        d = type("D", (), {})()
        d.index = 0
        d.backend_name = "CAP_DSHOW"
        d.width = w
        d.height = h
        d.fourcc_name = "YUY2"
        d.open_ms = 0.0
        d.fps = 0.0
        return d

    def closeEvent(self, event):
        """窗口关闭时确保释放摄像头。"""
        self.on_stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = DeviceCheckWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

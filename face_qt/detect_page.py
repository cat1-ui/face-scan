"""人脸检测页（考核主交付物）。

实时画面 + 检测框/关键点 + 右侧数值侧栏 + 控制台打印 + 30 秒多帧评测。

考核对应关系：
- 实时预览 / 实时检测 / 同帧多人脸全部检出 → 画面逐人绘制
- 检测质量实时判定 → 侧栏逐项达标着色 + 判定结论
- 数值展示（侧栏 + 控制台）→ _on_frame 同步 print
- 5 关键点参与姿态计算 → 两眼倾角、鼻尖偏移比（face_metrics）
- 多帧分位数判读 → 30 秒评测模式（P10/中位/P90 + 检出率）

线程模型：取帧 + 推理在 DetectWorker(QThread)，主线程仅做 UI 刷新，
避免窗口被系统标记「未响应」。
"""
import os
import time

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QFrame, QCheckBox, QGridLayout, QGroupBox
)

from core import config
from core.detector import FaceDetector
from camera_device import CameraDevice
import face_metrics as fm


class DetectWorker(QThread):
    """后台线程：读帧 + 检测 + 指标计算 + 绘制。

    信号 frame_ready 携带：
        canvas_bgr : 标注后的帧（BGR）
        metrics    : 本次检测的指标 dict（用于侧栏与控制台）
        fps        : 当前 FPS
        face_count : 本帧检出人脸数
    """
    frame_ready = pyqtSignal(object, object, float, int)

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
            t0 = time.time()
            ok, frame = self.device.read_frame()
            if not ok:
                continue

            # 检测（推理在工作线程）
            faces = self.detector.detect(frame)

            # 计算指标（含姿态，由 5 关键点推导）
            metrics = self._compute_metrics(frame, faces)

            # 绘制（框 + 5 点 + 置信度 + 尺寸条）
            self.detector.draw(frame, faces)
            if faces is not None and len(faces) >= 1:
                self.detector.draw_face_size_bar(frame, faces[0][2])

            face_count = 0 if faces is None else len(faces)
            single_frame_ms = (time.time() - t0) * 1000.0

            # FPS 统计（每秒更新一次）
            self._frames += 1
            now = time.time()
            if now - self._last >= 1.0:
                self._fps = self._frames / (now - self._last)
                self._frames = 0
                self._last = now

            metrics["fps"] = self._fps
            metrics["single_frame_ms"] = single_frame_ms
            metrics["face_count"] = face_count

            self.frame_ready.emit(frame, metrics, self._fps, face_count)

    def _compute_metrics(self, frame, faces):
        """对每帧计算可判读指标。多人时逐人计算。"""
        result = {"faces": []}
        if faces is None or len(faces) == 0:
            return result
        for fi, face in enumerate(faces):
            bbox = face[0:4]
            m = fm.compute_all(frame, bbox=bbox)
            tilt = fm.eye_tilt_angle(face)
            nose = fm.nose_offset_ratio(face)
            result["faces"].append({
                "index": fi,
                "score": float(face[14]),
                "face_width": m["face_width"],
                "eye_tilt": tilt,
                "nose_offset": nose,
                "brightness": m["brightness"],
                "sharpness": m["sharpness"],
            })
        return result

    def stop(self):
        self._running = False


class DetectPage(QWidget):
    def __init__(self):
        super().__init__()
        self.device = None
        self.worker = None
        self.detector = FaceDetector()
        self._last_canvas = None
        self._last_metrics = None

        # 30 秒评测累积器
        self._eval_active = False
        self._eval_samples = []

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        # 按钮行
        self.btn_start = QPushButton("打开摄像头")
        self.btn_stop = QPushButton("关闭摄像头")
        self.btn_stop.setEnabled(False)
        self.btn_upload = QPushButton("图片上传检测")
        self.btn_eval = QPushButton("30 秒评测")
        self.btn_eval.setEnabled(False)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_upload.clicked.connect(self.on_upload)
        self.btn_eval.clicked.connect(self.on_eval)

        self.chk_size_bar = QCheckBox("显示尺寸条")
        self.chk_size_bar.setChecked(True)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_upload)
        btn_row.addWidget(self.btn_eval)
        btn_row.addWidget(self.chk_size_bar)
        btn_row.addStretch()

        # 画面
        self.img_label = QLabel("点击「打开摄像头」开始实时检测")
        self.img_label.setFrameShape(QFrame.Shape.Box)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setMinimumSize(640, 420)

        # 侧栏数值面板
        side_panel = self._build_side_panel()

        # 底部状态栏
        self.status_label = QLabel("状态: 未启动")
        self.status_label.setStyleSheet("color: #555;")

        # 布局：左侧画面 + 右侧侧栏
        center = QHBoxLayout()
        center.addWidget(self.img_label, 4)
        center.addWidget(side_panel, 1)

        layout = QVBoxLayout()
        layout.addLayout(btn_row)
        layout.addLayout(center)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._refresh)
        self.ui_timer.setInterval(30)

    def _build_side_panel(self):
        """右侧数值侧栏：各项指标 + 达标着色。"""
        self.metric_labels = {}
        panel = QGroupBox("实时检测指标")
        grid = QGridLayout()

        rows = [
            ("face_count", "人脸数", ""),
            ("score", "置信度", "≥ %.2f" % config.DET_SCORE_THRESHOLD),
            ("face_width", "框宽(px)", "≥ %.0f" % config.FACE_WIDTH_MIN),
            ("eye_tilt", "两眼倾角(°)", "≤ %.0f" % config.EYE_TILT_MAX),
            ("nose_offset", "鼻尖偏移/脸宽", "≤ %.2f" % config.NOSE_OFFSET_MAX),
            ("brightness", "亮度", "%d~%d" % (config.BRIGHTNESS_MIN, config.BRIGHTNESS_MAX)),
            ("sharpness", "清晰度", "≥ %.0f" % config.SHARPNESS_MIN),
            ("fps", "处理帧率", "≥ 15"),
            ("single_frame_ms", "单帧耗时(ms)", "≤ 50"),
        ]
        for i, (key, label, _req) in enumerate(rows):
            name_lbl = QLabel(label)
            name_lbl.setStyleSheet("color: #555;")
            val_lbl = QLabel("--")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            val_lbl.setStyleSheet("font-weight: 600; color: #999;")
            self.metric_labels[key] = val_lbl
            grid.addWidget(name_lbl, i, 0)
            grid.addWidget(val_lbl, i, 1)

        self.eval_label = QLabel("评测: 未开始")
        self.eval_label.setStyleSheet("color: #555; margin-top: 8px;")
        self.eval_label.setWordWrap(True)

        outer = QVBoxLayout()
        outer.addLayout(grid)
        outer.addWidget(self.eval_label)
        outer.addStretch()
        panel.setLayout(outer)
        return panel

    def _set_metric(self, key, text, ok=None):
        """更新侧栏某项数值，并按达标与否着色。"""
        lbl = self.metric_labels.get(key)
        if lbl is None:
            return
        lbl.setText(text)
        if ok is None:
            lbl.setStyleSheet("font-weight: 600; color: #999;")
        elif ok:
            lbl.setStyleSheet("font-weight: 600; color: #1d9e75;")
        else:
            lbl.setStyleSheet("font-weight: 600; color: #e24b4a;")

    # ---------------- 摄像头 ----------------
    def on_start(self):
        self.btn_start.setEnabled(False)
        probe = CameraDevice(0)
        if not probe.open():
            self.status_label.setText("状态: 摄像头打开失败（被占用或不存在）")
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
            self.status_label.setText("状态: 打开失败")
            self.btn_start.setEnabled(True)
            return
        for _ in range(5):
            self.device.read_frame()

        self.worker = DetectWorker(self.device, self.detector)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.start()
        self.ui_timer.start()
        self.btn_stop.setEnabled(True)
        self.btn_eval.setEnabled(True)
        self.status_label.setText("状态: 检测中...")

    def _on_frame(self, canvas, metrics, fps, face_count):
        self._last_canvas = canvas
        self._last_metrics = metrics

        # 侧栏更新
        self._update_side_panel(metrics, fps, face_count)

        # 控制台打印（考核要求：数值同时打印到控制台）
        self._print_to_console(metrics, fps, face_count)

        # 30 秒评测采样
        if self._eval_active:
            self._eval_samples.append(self._snapshot(metrics, fps, face_count))

        self.status_label.setText(
            f"状态: 检测中 | FPS: {fps:.1f} | 人脸数: {face_count}")

    def _update_side_panel(self, metrics, fps, face_count):
        self._set_metric("face_count", str(face_count), face_count >= 1)
        self._set_metric("fps", f"{fps:.1f}", fps >= 15)

        faces = metrics.get("faces", [])
        if faces:
            f = faces[0]
            self._set_metric("score", f"{f['score']:.2f}",
                             f["score"] >= config.DET_SCORE_THRESHOLD)
            fw = f["face_width"]
            self._set_metric("face_width", f"{fw:.0f}" if fw else "--",
                             fw is not None and fw >= config.FACE_WIDTH_MIN)
            tilt = f["eye_tilt"]
            self._set_metric("eye_tilt", f"{tilt:.1f}" if tilt is not None else "--",
                             tilt is not None and tilt <= config.EYE_TILT_MAX)
            nose = f["nose_offset"]
            self._set_metric("nose_offset", f"{nose:.3f}" if nose is not None else "--",
                             nose is not None and nose <= config.NOSE_OFFSET_MAX)
            bright = f["brightness"]
            self._set_metric("brightness", f"{bright:.0f}",
                             config.BRIGHTNESS_MIN <= bright <= config.BRIGHTNESS_MAX)
            sharp = f["sharpness"]
            self._set_metric("sharpness", f"{sharp:.0f}",
                             sharp >= config.SHARPNESS_MIN)
            sf = metrics.get("single_frame_ms")
            self._set_metric("single_frame_ms", f"{sf:.1f}", sf <= 50)
        else:
            for k in ("score", "face_width", "eye_tilt", "nose_offset",
                      "brightness", "sharpness", "single_frame_ms"):
                self._set_metric(k, "--", None)

        if face_count >= 2:
            self.eval_label.setText(f"提示: 检测到 {face_count} 张脸（逐人框选，指标取第 1 张）")
        elif self._eval_active:
            self.eval_label.setText(f"评测中... 已采样 {len(self._eval_samples)} 帧")

    def _print_to_console(self, metrics, fps, face_count):
        faces = metrics.get("faces", [])
        sf = metrics.get("single_frame_ms")
        if not faces:
            print(f"[检测] FPS={fps:.1f} 单帧={sf:.1f}ms 人脸=0")
            return
        for f in faces:
            tilt = f["eye_tilt"]
            nose = f["nose_offset"]
            print(
                f"[检测] 置信度={f['score']:.2f} 框宽={f['face_width']:.0f}px "
                f"倾角={tilt:.1f}° 鼻偏={nose:.3f} "
                f"亮度={f['brightness']:.0f} 清晰度={f['sharpness']:.0f} "
                f"FPS={fps:.1f} 单帧={sf:.1f}ms"
            )

    @staticmethod
    def _snapshot(metrics, fps, face_count):
        faces = metrics.get("faces", [])
        if faces:
            f = faces[0]
            return {
                "score": f["score"],
                "face_width": f["face_width"] or 0,
                "eye_tilt": f["eye_tilt"],
                "nose_offset": f["nose_offset"],
                "brightness": f["brightness"],
                "sharpness": f["sharpness"],
                "fps": fps,
                "single_frame_ms": metrics.get("single_frame_ms", 0),
            }
        return None

    # ---------------- 30 秒评测 ----------------
    def on_eval(self):
        if self._eval_active:
            return
        self._eval_active = True
        self._eval_samples = []
        self.eval_label.setText("评测中... 请保持人脸在画面内 30 秒")
        QTimer.singleShot(30000, self._finish_eval)

    def _finish_eval(self):
        self._eval_active = False
        samples = [s for s in self._eval_samples if s is not None]
        total = len(self._eval_samples)
        if not samples:
            self.eval_label.setText("评测失败: 未采到有人脸的帧")
            print("[评测] 失败: 无有效样本")
            return

        def pct(key, p):
            vals = [s[key] for s in samples if s.get(key) is not None]
            if not vals:
                return None
            return float(np.percentile(vals, p))

        lines = ["【30 秒评测结果】",
                 f"  总采样帧: {total}  有效帧(有人脸): {len(samples)}"]
        if total:
            lines.append(f"  检出率: {len(samples)/total:.1%}")

        for key, name in [
            ("score", "置信度"), ("face_width", "框宽"),
            ("eye_tilt", "倾角"), ("nose_offset", "鼻偏"),
            ("brightness", "亮度"), ("sharpness", "清晰度"),
            ("fps", "帧率"), ("single_frame_ms", "单帧耗时"),
        ]:
            p10, p50, p90 = pct(key, 10), pct(key, 50), pct(key, 90)
            if p50 is None:
                continue
            lines.append(f"  {name}: P10={p10:.2f} 中位={p50:.2f} P90={p90:.2f}")

        text = "\n".join(lines)
        print("\n" + text + "\n")
        self.eval_label.setText(text)

    # ---------------- 刷新 ----------------
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
        self._eval_active = False
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
        self.btn_eval.setEnabled(False)
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
            self.status_label.setText("状态: 图片读取失败")
            return
        try:
            if img.size == 0 or img.shape[0] < 20 or img.shape[1] < 20:
                self.status_label.setText("状态: 图片尺寸过小")
                return
            faces = self.detector.detect(img)
            canvas = self.detector.draw(img.copy(), faces)
            n = 0 if faces is None else len(faces)
            self.status_label.setText(f"状态: 图片检测完成 | 人脸数: {n}")
            h, w = canvas.shape[:2]
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.img_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        except cv2.error as e:
            self.status_label.setText(f"状态: 检测异常: {e}")

    def hideEvent(self, event):
        self.on_stop()
        super().hideEvent(event)

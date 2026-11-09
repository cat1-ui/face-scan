"""人脸识别系统 · PyQt6 主应用。

Tab 结构：设备体检 / 人脸检测 / 人脸录入 / 人脸识别。
摄像头由各页面自行管理（同一时刻只有一个页面在使用）。

运行：.venv/Scripts/python.exe app.py
"""
import sys

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel,
    QPushButton
)
from PyQt6.QtCore import Qt

from core import config


class PlaceholderPage(QWidget):
    """占位页（M3/M4 未实现时显示）。"""

    def __init__(self, title, desc):
        super().__init__()
        layout = QVBoxLayout()
        layout.addStretch()
        t = QLabel(title)
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet("font-size: 20px; color: #888;")
        d = QLabel(desc)
        d.setAlignment(Qt.AlignmentFlag.AlignCenter)
        d.setStyleSheet("color: #aaa;")
        layout.addWidget(t)
        layout.addWidget(d)
        layout.addStretch()
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸识别系统")
        self.resize(1000, 700)

        tabs = QTabWidget()
        tabs.addTab(self._build_device_check_tab(), "设备体检")
        tabs.addTab(self._build_detect_tab(), "人脸检测")
        tabs.addTab(self._build_enroll_tab(), "人脸录入")
        tabs.addTab(self._build_attend_tab(), "人脸识别")
        self.setCentralWidget(tabs)

    def _build_device_check_tab(self):
        """设备体检页：直接嵌入 M1 已验证的 DeviceCheckWindow。"""
        from device_check_ui import DeviceCheckWindow
        w = DeviceCheckWindow()
        # 作为 Tab 内容，去掉独立窗口标题栏的影响（无需额外处理）
        return w

    def _build_detect_tab(self):
        """人脸检测页（M2）。"""
        from detect_page import DetectPage
        return DetectPage()

    def _build_enroll_tab(self):
        """人脸录入页（M3）。"""
        from enroll_page import EnrollPage
        return EnrollPage()

    def _build_attend_tab(self):
        """人脸识别页（M4）。"""
        from attend_page import AttendPage
        return AttendPage()


def main():
    config.ensure_dirs()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

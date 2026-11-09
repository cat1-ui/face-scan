"""下载 YuNet + SFace 模型到 models/ 目录。

注：本项目模型已由用户手动提供（在桌面），本脚本作为兜底/换机复现用。
如需重新下载，模型官方地址如下（国内访问可能需镜像/手动下载）：

- YuNet  (face_detection_yunet_2023mar.onnx):
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
- SFace  (face_recognition_sface_2021dec.onnx):
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

运行：.venv/Scripts/python.exe tools/download_models.py
"""
import os
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, "models")

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}


def download():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, url in MODELS.items():
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"[跳过] {name} 已存在 ({os.path.getsize(path)} 字节)")
            continue
        print(f"[下载] {name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"      完成 ({os.path.getsize(path)} 字节)")


if __name__ == "__main__":
    download()

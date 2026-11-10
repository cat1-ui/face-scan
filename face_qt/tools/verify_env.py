"""环境验证脚本：确认 YuNet 能加载、能检测，环境依赖齐全。

严格对应《人脸识别系统_实施方案》M0 验收标准 A0-1 ~ A0-7。
运行：.venv/Scripts/python.exe tools/verify_env.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # face_qt/
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "face_data")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")

# 两个模型文件的真实文件名
DET_MODEL = "face_detection_yunet_2023mar.onnx"
REC_MODEL = "face_recognition_sface_2021dec.onnx"

# 一张用于验证的测试图（若存在则做真实检测）
TEST_IMG = os.path.join(BASE_DIR, "tools", "test_face.jpg")

results = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f"  -> {detail}" if detail else ""))


def main():
    print("=" * 60)
    print("人脸识别系统 · 环境验证 (M0)")
    print("=" * 60)

    # A0-1 cv2 版本
    try:
        import cv2
        check("A0-1 cv2 版本", cv2.__version__ >= "4.5.4",
              f"cv2.__version__ = {cv2.__version__}")
    except Exception as e:
        check("A0-1 cv2 版本", False, f"导入失败: {e}")

    # A0-2 两个 onnx 存在且大小正确
    for name in (DET_MODEL, REC_MODEL):
        p = os.path.join(MODELS_DIR, name)
        ok = os.path.exists(p)
        size = os.path.getsize(p) if ok else 0
        check(f"A0-2 模型存在 {name}", ok and size > 0, f"{size} 字节")

    # A0-3 FaceDetectorYN
    check("A0-3 cv2.FaceDetectorYN", hasattr(cv2, "FaceDetectorYN"))

    # A0-4 FaceRecognizerSF
    check("A0-4 cv2.FaceRecognizerSF", hasattr(cv2, "FaceRecognizerSF"))

    # A0-5 PyQt6
    try:
        from PyQt6.QtWidgets import QApplication  # noqa: F401
        check("A0-5 PyQt6 可导入", True)
    except Exception as e:
        check("A0-5 PyQt6 可导入", False, f"{e}")

    # A0-6 数据库 4 张表 + WAL
    import sqlite3
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        conn.close()
        expected = {"person", "face_feature", "face_photo", "attendance"}
        check("A0-6 数据库 4 张表 + WAL",
              expected.issubset(set(tables)) and journal == "wal",
              f"tables={tables}, journal={journal}")
    else:
        check("A0-6 数据库 4 张表 + WAL", False, "数据库不存在，先运行 init_db.py")

    # A0-7 真实检测：能加载 YuNet 并在测试图上框出人脸
    if os.path.exists(TEST_IMG):
        try:
            img = cv2.imread(TEST_IMG)
            det = cv2.FaceDetectorYN.create(
                os.path.join(MODELS_DIR, DET_MODEL), "", (320, 320), 0.9)
            h, w = img.shape[:2]
            det.setInputSize((w, h))
            _, faces = det.detect(img)
            n = 0 if faces is None else len(faces)
            check("A0-7 模型能检测到人脸", n >= 1, f"检测到 {n} 张脸")
        except Exception as e:
            check("A0-7 模型能检测到人脸", False, f"{e}")
    else:
        check("A0-7 模型能检测到人脸", None,
              f"未找到测试图 {TEST_IMG}，跳过（可放一张人脸图到 tools/test_face.jpg 后重跑）")

    # 汇总
    print("=" * 60)
    fails = [r for r in results if r[1] == "FAIL"]
    passes = [r for r in results if r[1] == "PASS"]
    print(f"通过 {len(passes)} 项，失败 {len(fails)} 项")
    if fails:
        print("失败项:")
        for name, _, detail in fails:
            print(f"  - {name}: {detail}")
    print("=" * 60)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())

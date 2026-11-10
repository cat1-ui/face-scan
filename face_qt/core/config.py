"""全局配置。

集中管理路径、模型文件名、以及 M1 实测推导出的阈值/准入判据。
M1 阶段会把「建议阈值」写入这里，M3 质量门控直接复用。
"""
import os

# ---- 路径 ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # face_qt/
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "face_data")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
PHOTOS_DIR = os.path.join(DATA_DIR, "photos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ---- 模型文件（真实文件名）----
DET_MODEL = "face_detection_yunet_2023mar.onnx"
REC_MODEL = "face_recognition_sface_2021dec.onnx"

# ---- 人脸检测 ----
DET_SCORE_THRESHOLD = 0.85         # YuNet 检测置信度阈值（考核要求 ≥ 0.85）
ALIGN_SIZE = 112                    # SFace 对齐裁剪尺寸

# ---- 特征 ----
FEATURE_DIM = 128                   # SFace 输出维度
FEATURE_DTYPE = "float32"

# ---- 质量门控阈值 ----
# 清晰度：人脸 ROI 归一化 112×112 后的拉普拉斯方差（加 3×3 高斯平滑）。
# 考核要求：不低于 60。
SHARPNESS_MIN = 60.0                 # 清晰度门槛（拉普拉斯方差）
BRIGHTNESS_MIN = 55.0               # 亮度下限
BRIGHTNESS_MAX = 215.0              # 亮度上限
FACE_WIDTH_MIN = 80.0               # 人脸最小宽度（像素）

# ---- 姿态判据（考核要求，由 5 关键点推导）----
EYE_TILT_MAX = 20.0                 # 两眼连线倾角上限（度）
NOSE_OFFSET_MAX = 0.2               # 鼻尖横向偏移 / 脸宽 上限

# ---- 识别 ----
SIMILARITY_THRESHOLD = 0.5        # 出厂阈值（M5 会实测定档覆盖）
GAP_THRESHOLD = 0.08                # top1-top2 相似脸判定间隔

# ---- 录入 ----
DUP_THRESHOLD = 0.92                # 相似度去重阈值
MAX_SAMPLES_PER_PERSON = 5          # 每人样本上限


def ensure_dirs():
    for d in (MODELS_DIR, DATA_DIR, PHOTOS_DIR, OUTPUT_DIR):
        os.makedirs(d, exist_ok=True)

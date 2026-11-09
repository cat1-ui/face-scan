# 人脸采集质检工具

刷脸考勤的配套工具：先把「这张脸能不能被识别人」查清楚，再谈识别率。
PyQt6 桌面端（4 个页签）+ FastAPI 报表端，纯 CPU 运行，无 GPU 依赖。

## 为什么做这个

人脸识别项目里，识别率上不去往往不是模型的问题，而是采集端的问题：光太暗、脸太小、画面糊、脸库里有重复样本。
这个工具把「设备能力」和「图像质量」量化成可判定的指标，再用正负样本的分数分布把识别阈值定下来，而不是拍脑袋写一个 0.5。

## 功能

| 页签 | 说明 |
|---|---|
| 设备体检 | 摄像头可用性、分辨率、光线条件检查，输出体检报告 |
| 人脸检测 | 实时框出人脸，展示清晰度、亮度、人脸宽度等指标 |
| 人脸录入 | 质量门控通过才允许保存；相似度去重阈值 0.92；每人样本上限 5 张 |
| 人脸识别 | 实时比对，top1 与 top2 间隔小于 0.08 时判为「不确定」而不是硬认 |

报表端（`face_web`，11 个接口）：考勤记录查询、人员管理、人脸样本浏览、Excel 导出。

## 阈值是怎么定的（这个项目的重点）

完整过程见 [`face_qt/阈值定档记录.md`](face_qt/阈值定档记录.md)：五连测采集正负样本，比较分数分布。

| 指标 | 数值 |
|---|---|
| 正样本（同人，n=12）中位 | 0.940 |
| 负样本（不同人，n=9）中位 | 0.321 |
| 可分性间隔 | 0.549（硬性门槛 ≥ 0.25，达标） |

结论：定档 **0.50**（出厂默认 0.363 偏松，会减少拒识但放大误识）。采集设备为笔记本内置摄像头，硬件上限 1280×720。

质量门控阈值（`face_qt/core/config.py`）：清晰度用拉普拉斯方差，门槛 50（本机实测中位约 70、P25 约 66）；亮度 55–215；人脸最小宽度 80px。

## 技术栈

- 桌面端：PyQt6 · OpenCV（YuNet 检测 + SFace 识别，纯 CPU） · SQLite
- 报表端：FastAPI + SQLite；前端 Vue + Vite
- 无第三方人脸 SDK，模型直接跑 OpenCV 的 `FaceDetectorYN` / `FaceRecognizerSF`

规模：Python 3561 行（含桌面端与报表端）。

## 快速开始

### 1. 准备模型（不入库）

两个 onnx 模型体积较大，未纳入版本管理，用脚本获取：

```bash
cd face_qt
python tools/download_models.py
```

或按脚本注释里的官方地址手动下载，放进 `face_qt/models/`：
`face_detection_yunet_2023mar.onnx`、`face_recognition_sface_2021dec.onnx`

### 2. 桌面端

```bash
cd face_qt
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python app.py
```

### 3. 报表端

```bash
cd face_web
pip install -r requirements.txt
uvicorn main:app --port 8000
```

前端在 `face_web/frontend`：`npm install && npm run dev`（默认 5173，接口地址写在组件里指向 `http://127.0.0.1:8000`）。

> 桌面端和报表端共用同一个 `face_data/attendance.db`，先跑桌面端录入数据，报表端才有内容可看。

## 数据与隐私

仓库里**不含任何人脸照片、考勤库和模型权重**：`face_data/`（`attendance.db` 与 `photos/`）、`face_qt/output/`（导出的质检报告）、`face_qt/models/` 全部在 `.gitignore` 中。克隆后首次运行会自动建空库。

## 安全说明

- 已修：`GET /api/photos/{path:path}` 存在目录穿越 —— 原来直接把请求路径拼进 `os.path.join`，`../` 可以读到本机任意文件；现在先 `abspath` 归一化，再校验结果必须落在 `photos/` 目录内，跨盘符也会被拦
- 已修：CORS 的 `allow_origins=["*"]` + `allow_credentials=True` 组合 → 改为白名单（默认本机 5173）+ 关闭凭据
- 报表端**没有鉴权**，仅限本机使用

## 已知限制

- 需要可用摄像头，无摄像头时只有「设备体检」页能给出明确失败原因
- 单人脸场景；不处理多人同框与小目标
- 阈值基于一台笔记本摄像头的实测数据，换设备建议重新跑一遍定档流程

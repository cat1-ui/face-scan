"""体检报告生成。

职责（实施方案 M1 T1-6 ~ T1-8）：
- 采样累积统计：P10 / 中位 / P90
- 准入判据推导：从实测 P10 推导建议阈值
- 生成报告 + 导出 output/camera_report_*.txt

质量问题必须可执行：每条问题带「→ 处置」，不同病因给不同建议。
"""
import os
import datetime

import numpy as np

from core.config import OUTPUT_DIR, ensure_dirs


def percentile(arr, p):
    """分位数。arr 为一维数组/列表，p 为 0~100。"""
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return None
    return float(np.percentile(a, p))


def summarize(samples, key):
    """对 samples 列表中的某指标算 P10/中位/P90。"""
    values = [s[key] for s in samples if s.get(key) is not None]
    if not values:
        return {"count": 0, "p10": None, "p50": None, "p90": None}
    return {
        "count": len(values),
        "p10": percentile(values, 10),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
    }


def derive_thresholds(samples):
    """从实测 P10 推导建议阈值（准入判据）。

    返回 dict，含各指标建议阈值。原则：取 P10 作为下限参考，
    即 90% 的帧都优于该值。
    """
    th = {}
    for key in ("sharpness", "brightness", "contrast"):
        s = summarize(samples, key)
        th[key] = s["p10"]
    return th


def rate_quality(summary):
    """综合评级 A~D。基于清晰度和亮度的中位值粗略判断。

    返回 (等级, 说明)。
    """
    sharp = summary.get("sharpness", {})
    bright = summary.get("brightness", {})
    sp50 = sharp.get("p50")
    bp50 = bright.get("p50")
    if sp50 is None or bp50 is None:
        return "D", "样本不足，无法评级"

    issues = 0
    if sp50 < 60:
        issues += 1
    if bp50 < 55 or bp50 > 215:
        issues += 1

    if issues == 0:
        return "A", "画质基线良好"
    if issues == 1:
        return "B", "存在一项画质问题"
    return "C", "画质基线偏弱，建议改善光照或设备"


def build_problems(samples, summary):
    """按影响排序生成问题清单，每条带可执行处置。"""
    problems = []
    sharp = summary.get("sharpness", {})
    bright = summary.get("brightness", {})

    bp50 = bright.get("p50")
    sp50 = sharp.get("p50")

    # 亮度问题（前置判断）
    if bp50 is not None and bp50 < 55:
        problems.append({
            "问题": "光照不足",
            "数值": f"亮度中位 {bp50:.1f}",
            "处置": "开灯 / 转向光源 / 换位置",
        })
    elif bp50 is not None and bp50 > 215:
        problems.append({
            "问题": "画面过曝 / 强逆光",
            "数值": f"亮度中位 {bp50:.1f}",
            "处置": "背对光源",
        })

    # 亮度正常但清晰度低 → 模糊
    if sp50 is not None and sp50 < 60 and (bp50 is None or 55 <= bp50 <= 215):
        problems.append({
            "问题": "画面模糊",
            "数值": f"清晰度中位 {sp50:.1f}",
            "处置": "稳住 / 擦净镜头 / 换摄像头",
        })

    # 脸宽
    fw = summary.get("face_width", {})
    if fw.get("p50") is not None and fw["p50"] < 80:
        problems.append({
            "问题": "人脸过小",
            "数值": f"脸宽中位 {fw['p50']:.0f}px",
            "处置": "靠近摄像头 / 提高分辨率",
        })

    if not problems:
        problems.append({
            "问题": "无明显问题",
            "数值": "各指标均在正常区间",
            "处置": "保持当前采集环境",
        })
    return problems


def generate_report(device, samples, resolution_records, codec_records,
                    total_frames, valid_frames, detect_rate, duration_s):
    """生成完整报告 dict，并返回。"""
    summary = {
        "sharpness": summarize(samples, "sharpness"),
        "brightness": summarize(samples, "brightness"),
        "contrast": summarize(samples, "contrast"),
        "color_temp": summarize(samples, "color_temp"),
        "face_width": summarize(samples, "face_width"),
    }
    thresholds = derive_thresholds(samples)
    level, level_desc = rate_quality(summary)
    problems = build_problems(samples, summary)
    hw_max = None
    if resolution_records:
        from camera_device import CameraDevice
        hw_max = CameraDevice.hardware_max_resolution(
            CameraDevice.__new__(CameraDevice), resolution_records)

    report = {
        "采样概况": {
            "总帧数": total_frames,
            "有效帧": valid_frames,
            "检出率": f"{detect_rate:.1%}" if total_frames else "N/A",
            "采样时长(秒)": round(duration_s, 1),
        },
        "设备": {
            "index": device.index,
            "后端": device.backend_name,
            "实际分辨率": f"{device.width}x{device.height}",
            "帧率": device.fps if getattr(device, "fps", None) else "N/A",
            "编码": device.fourcc_name,
            "打开耗时(ms)": device.open_ms,
        },
        "分辨率探测过程": resolution_records,
        "硬件上限": f"{hw_max[0]}x{hw_max[1]}" if hw_max else "未知",
        "图像质量": summary,
        "建议阈值": thresholds,
        "综合评级": f"{level}（{level_desc}）",
        "主要问题": problems,
        "下一步": [
            "确认设备体检通过后，进入 M2 人脸检测",
            "将建议阈值已写入 core/config.py，M3 门控直接复用",
        ],
    }
    return report


def render_text(report):
    """把报告 dict 渲染成可读文本。"""
    lines = []
    lines.append("=" * 60)
    lines.append("人脸识别系统 · 设备体检报告")
    lines.append("=" * 60)

    def fmt_value(v):
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    for section in ("采样概况", "设备", "图像质量", "建议阈值", "综合评级"):
        lines.append(f"\n【{section}】")
        val = report[section]
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, dict):
                    lines.append(f"  {k}: p10={fmt_value(v.get('p10'))} "
                                 f"中位={fmt_value(v.get('p50'))} "
                                 f"p90={fmt_value(v.get('p90'))} "
                                 f"(n={v.get('count')})")
                else:
                    lines.append(f"  {k}: {fmt_value(v)}")
        else:
            lines.append(f"  {fmt_value(val)}")

    lines.append("\n【分辨率探测过程】")
    for r in report["分辨率探测过程"]:
        mark = "OK " if r["ok"] else "降级"
        lines.append(f"  {r['resolution']}: 读回 {r['width_actual']}x"
                     f"{r['height_actual']} [{mark}]")
    lines.append(f"  硬件上限: {report['硬件上限']}")

    lines.append("\n【主要问题】")
    for p in report["主要问题"]:
        lines.append(f"  - {p['问题']}（{p['数值']}）→ {p['处置']}")

    lines.append("\n【下一步】")
    for a in report["下一步"]:
        lines.append(f"  - {a}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def export_report(report):
    """导出报告到 output/camera_report_*.txt，返回文件路径。"""
    ensure_dirs()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"camera_report_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_text(report))
    return path

"""设备体检主流程（命令行版）。

串起：设备探测 → 循环采样 → 统计 → 阈值推导 → 报告生成 → 阈值写回 config.py。

运行：.venv/Scripts/python.exe device_check.py
采样期间按 Ctrl+C 或达到最大帧数自动停止。
"""
import sys
import time

import cv2

import face_metrics as fm
from camera_device import CameraDevice
import report as rp
from core import config


# 采样参数
MAX_FRAMES = 150            # 最多采样帧数
SAMPLE_INTERVAL = 0.1       # 采样间隔（秒），约 10 FPS 采样


def run():
    print("=" * 60)
    print("人脸识别系统 · 设备体检 (M1)")
    print("=" * 60)

    # 1. 探测设备（后端 + 分辨率 + 编码）
    device = CameraDevice(index=0)
    print("\n[1/3] 探测设备 ...")
    if not device.open():
        print("❌ 摄像头打开失败：可能被占用或不存在。")
        print("   排查：关闭微信/钉钉/Zoom；检查设备管理器；换 index。")
        return 1

    print(f"  后端: {device.backend_name}")
    print(f"  默认分辨率: {device.width}x{device.height}")
    print(f"  打开耗时: {device.open_ms} ms")
    device.release()

    # 分辨率探测（复用句柄，快）
    print("\n[2/3] 分辨率探测 ...")
    res_records = device.probe_resolutions()
    for r in res_records:
        mark = "OK " if r["ok"] else "降级"
        print(f"  {r['resolution']:>10} -> 读回 {r['width_actual']}x"
              f"{r['height_actual']} [{mark}]")
    hw = device.hardware_max_resolution(res_records)
    if hw:
        print(f"  硬件上限: {hw[0]}x{hw[1]}")
    else:
        print("  硬件上限: 未知")

    # 编码探测（复用句柄）
    codec_records = device.probe_codecs(hw[0], hw[1]) if hw else []
    for c in codec_records:
        mark = "OK" if c["ok"] else f"不可用(实际 {c.get('got')})"
        print(f"  编码 {c['codec']}: {mark}")

    # 用硬件上限分辨率重新打开，用于采样
    print(f"\n[3/3] 以 {hw[0]}x{hw[1]} 重新打开并采样 ...")
    if not device.open(width=hw[0], height=hw[1]):
        print("❌ 以目标分辨率打开失败")
        return 1
    print(f"  实际分辨率: {device.width}x{device.height}")
    print(f"  编码: {device.fourcc_name}")
    print("  请正对摄像头，保持自然（约 10 秒，按 Ctrl+C 可提前结束）")

    samples = []
    total_frames = 0
    valid_frames = 0
    start = time.time()
    last_tick = time.time()
    read_start = time.time()   # 实测帧率：只统计读帧耗时

    det = cv2.FaceDetectorYN.create(
        f"{config.MODELS_DIR}/{config.DET_MODEL}", "", (320, 320), 0.9)

    try:
        while total_frames < MAX_FRAMES:
            ok, frame = device.read_frame()
            if not ok:
                time.sleep(0.02)
                continue
            total_frames += 1

            # 人脸检测，只统计有人脸的帧
            h, w = frame.shape[:2]
            det.setInputSize((w, h))
            _, faces = det.detect(frame)

            if faces is not None and len(faces) >= 1:
                valid_frames += 1
                bbox = faces[0][0:4]
                m = fm.compute_all(frame, bbox=bbox)
                samples.append(m)

            # 采样间隔
            now = time.time()
            if now - last_tick >= SAMPLE_INTERVAL:
                last_tick = now
                print(f"\r  已采样 {total_frames} 帧，有效(有人脸) {valid_frames} 帧",
                      end="", flush=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n  采样被手动终止")

    duration = time.time() - start
    read_duration = time.time() - read_start
    fps = total_frames / read_duration if read_duration > 0 else 0
    device.fps = round(fps, 1)
    device.release()
    print(f"\n  采样结束：共 {total_frames} 帧，有效 {valid_frames} 帧，"
          f"时长 {duration:.1f}s")
    print(f"  实测帧率: {device.fps} FPS")
    print("  摄像头已释放")

    if valid_frames == 0:
        print("❌ 没有采到任何有人脸的帧，无法生成有效报告。")
        print("   请确认摄像头前有人脸，重跑。")
        return 1

    # 4. 生成报告 + 导出
    detect_rate = valid_frames / total_frames if total_frames else 0
    report = rp.generate_report(
        device, samples, res_records, codec_records,
        total_frames, valid_frames, detect_rate, duration)

    txt = rp.render_text(report)
    print("\n" + txt)

    path = rp.export_report(report)
    print(f"\n报告已导出: {path}")

    # 5. 阈值写回 config.py
    write_thresholds(report["建议阈值"])

    return 0


def write_thresholds(thresholds):
    """把建议阈值写回 core/config.py（M1 硬性门槛 A1-8）。"""
    cfg_path = config.__file__
    with open(cfg_path, "r", encoding="utf-8") as f:
        content = f.read()

    updated = False
    val = thresholds.get("sharpness")
    if val is not None:
        old = f"SHARPNESS_MIN = {config.SHARPNESS_MIN}"
        new = f"SHARPNESS_MIN = {val:.1f}"
        if old in content:
            content = content.replace(old, new)
            updated = True

    if updated:
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ 建议阈值已写入 core/config.py")
        print(f"  清晰度阈值: SHARPNESS_MIN = {val:.1f}")
    else:
        print("  未更新阈值（清晰度样本不足或无变化）")


if __name__ == "__main__":
    sys.exit(run())

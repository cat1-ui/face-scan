"""五连测诊断脚本（M5）。

用数据回答「这套画质+算法组合可分性够不够」。

五连测：
  ① 同人同姿态 × 连续 30 帧
  ② 同人转头 15°
  ③ 同人距离变化（近/远）
  ④ 同人光照变化（开灯/关灯）
  ⑤ 不同人两两组合

采集流程（引导式）：
  1. 录入 A 同学（正脸 3 张）
  2. 录入 B 同学（正脸 3 张）
  3. 依次引导采集五组数据，每组算相似度
  4. 统计正/负样本分布 → 找谷底 → 定档

运行：.venv/Scripts/python.exe tools/diagnose.py
"""
import os
import sys
import time
import csv
import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from camera_device import CameraDevice


# 测试配置
FRAMES_SAME_POSE = 30       # ①同姿态帧数
FRAMES_VARIANT = 20         # ②③④变体帧数
SAMPLE_INTERVAL = 0.3       # 采样间隔（秒），放慢给转体/换动作留时间


def _grab_frames(device, detector, n, interval=SAMPLE_INTERVAL):
    """连续抓 n 帧有人脸的图，返回 (frames, faces) 列表。"""
    frames = []
    faces_list = []
    while len(frames) < n:
        ok, frame = device.read_frame()
        if not ok:
            continue
        faces = detector.detect(frame)
        if faces is None or len(faces) != 1:
            continue
        frames.append(frame)
        faces_list.append(faces[0])
        time.sleep(interval)
    return frames, faces_list


def _features_of(recognizer, frames, faces_list):
    """对帧列表提取特征，返回 (n, 128) 数组。"""
    feats = []
    for frame, face in zip(frames, faces_list):
        _, f = recognizer.align_and_feature(frame, face)
        feats.append(f)
    return np.stack(feats)


def _pairwise_sim(feats_a, feats_b):
    """两组特征两两算余弦相似度，返回数组。"""
    return feats_a @ feats_b.T


def _self_sim(feats):
    """同组内两两相似度（去掉对角线）。"""
    sims = feats @ feats.T
    n = sims.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return sims[mask]


class DiagnoseRunner:
    def __init__(self):
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()

    def enroll_person(self, device, name, n_frames=3):
        """录入一个人，返回其参考特征（n_frames, 128）。"""
        print(f"\n>>> 录入【{name}】")
        print(f"    ⚠️ 镜头前只能有 {name} 一个人，其他人请完全离开画面！")
        self._countdown(6, f"    {name} 就位倒计时")
        frames, faces = _grab_frames(device, self.detector, n_frames)
        feats = _features_of(self.recognizer, frames, faces)
        print(f"    已采集 {n_frames} 帧，特征提取完成")
        return feats

    def _countdown(self, seconds, label):
        """倒计时提示。"""
        for i in range(seconds, 0, -1):
            print(f"\r    {label}: {i} 秒", end="", flush=True)
            time.sleep(1)
        print()

    def run(self):
        print("=" * 60)
        print("M5 五连测 · 可分性诊断")
        print("=" * 60)
        print("\n本测试需要两位真人（A 和 B）。")
        print("每个环节前有倒计时，请在倒计时内就位。\n")

        device = CameraDevice(0)
        device.open(width=1280, height=720)
        for _ in range(5):
            device.read_frame()

        results = {}   # 各测试组的相似度数据

        # 先录入 A、B 两人
        feats_a = self.enroll_person(device, "A 同学")
        feats_b = self.enroll_person(device, "B 同学")

        # ① 同人同姿态（A）
        print("\n>>> ① 同人同姿态：请 A 同学保持不动")
        self._countdown(5, "    就位倒计时")
        frames, faces = _grab_frames(device, self.detector, FRAMES_SAME_POSE)
        f_a_pose = _features_of(self.recognizer, frames, faces)
        results["①同姿态"] = _self_sim(f_a_pose)

        # ② 同人转头（A）
        print("\n>>> ② 同人转头 15°：请 A 同学轻轻左右转头")
        self._countdown(5, "    就位倒计时")
        frames, faces = _grab_frames(device, self.detector, FRAMES_VARIANT)
        f_a_turn = _features_of(self.recognizer, frames, faces)
        results["②转头"] = _pairwise_sim(f_a_turn, f_a_pose).flatten()

        # ③ 同人距离变化（A）
        print("\n>>> ③ 同人距离变化：请 A 同学先靠近再后退")
        self._countdown(5, "    就位倒计时")
        frames, faces = _grab_frames(device, self.detector, FRAMES_VARIANT)
        f_a_dist = _features_of(self.recognizer, frames, faces)
        results["③距离"] = _pairwise_sim(f_a_dist, f_a_pose).flatten()

        # ④ 同人光照变化（A）
        print("\n>>> ④ 同人光照变化：请改变光照（开灯/关灯）")
        self._countdown(5, "    就位倒计时")
        frames, faces = _grab_frames(device, self.detector, FRAMES_VARIANT)
        f_a_light = _features_of(self.recognizer, frames, faces)
        results["④光照"] = _pairwise_sim(f_a_light, f_a_pose).flatten()

        # ⑤ 不同人两两组合（A vs B）
        print("\n>>> ⑤ 不同人组合：请 B 同学正对摄像头")
        print("    ⚠️ A 同学请完全离开镜头！镜头前只留 B 一人！")
        self._countdown(8, "    B 就位倒计时（A 请离开）")
        frames, faces = _grab_frames(device, self.detector, FRAMES_VARIANT)
        f_b = _features_of(self.recognizer, frames, faces)
        results["⑤不同人"] = _pairwise_sim(f_b, feats_a).flatten()

        device.release()
        print("\n采集完成，摄像头已释放")

        # 统计
        self._summarize(results)

    def _summarize(self, results):
        print("\n" + "=" * 60)
        print("五连测统计结果")
        print("=" * 60)

        pos_sims = []   # 正样本（同人）
        neg_sims = []   # 负样本（不同人）

        for name, sims in results.items():
            s = np.asarray(sims)
            print(f"\n【{name}】 n={s.size}")
            print(f"  均值 {s.mean():.3f}  中位 {np.median(s):.3f}  "
                  f"P10 {np.percentile(s,10):.3f}  P90 {np.percentile(s,90):.3f}")
            if name == "⑤不同人":
                neg_sims.extend(s.tolist())
            else:
                pos_sims.extend(s.tolist())

        # 可分性判断
        print("\n" + "=" * 60)
        print("可分性判断")
        print("=" * 60)
        if pos_sims and neg_sims:
            pos_min = np.min(pos_sims)
            neg_max = np.max(neg_sims)
            gap = pos_min - neg_max
            print(f"正样本最小相似度: {pos_min:.3f}")
            print(f"负样本最大相似度: {neg_max:.3f}")
            print(f"可分性间隔: {gap:.3f}")
            print(f"判断: {'✅ 可分性成立（≥0.25）' if gap >= 0.25 else '❌ 可分性不足（<0.25）'}")

            # 候选阈值 = 两峰谷底（简化为正负样本的中值）
            candidate = (np.median(pos_sims) + np.median(neg_sims)) / 2
            print(f"\n候选阈值（谷底）: {candidate:.3f}")
            print(f"出厂阈值: {config.SIMILARITY_THRESHOLD}")

            # 导出 CSV
            self._export(results, pos_sims, neg_sims, candidate, gap)

    def _export(self, results, pos_sims, neg_sims, candidate, gap):
        ensure = getattr(config, "ensure_dirs", None)
        if ensure:
            ensure()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.OUTPUT_DIR, f"diagnose_{ts}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "similarity"])
            for s in pos_sims:
                w.writerow(["positive", round(s, 6)])
            for s in neg_sims:
                w.writerow(["negative", round(s, 6)])
        print(f"\n相似度分布已导出: {path}")
        print(f"候选阈值: {candidate:.3f}（请人工确认后写入 config.py 定档）")


def main():
    runner = DiagnoseRunner()
    runner.run()


if __name__ == "__main__":
    main()

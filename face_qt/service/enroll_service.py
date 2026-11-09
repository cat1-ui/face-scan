"""录入编排逻辑（service 层）。

核心职责（实施方案 M3）：
- 指令驱动采集：3 条指令（正脸/左转/右转），配额 3/1/1
- 相似度去重（阈值 0.92）
- 采集即入库（不是攒到最后）
- 每人样本上限 5 张
- 重复学号拒绝
- 多人画面拦截（≥2 张脸拒绝，在 quality_gate G0 已处理）

本模块是「纯逻辑」，不直接操作摄像头/UI，
便于用 tools/test_ui_logic.py 做回调级测试（A3-19）。
"""
import datetime
import os

import numpy as np

from core import config
from repository.person_repo import PersonRepo
from repository.feature_repo import FeatureRepo


# 采集指令定义
INSTRUCTIONS = [
    {"index": 0, "text": "请正对摄像头，保持自然表情（保持不动 3 秒）",
     "grab": 12, "quota": 3},
    {"index": 1, "text": "请把头轻轻向左转一下（约 15°）",
     "grab": 8, "quota": 1},
    {"index": 2, "text": "请把头轻轻向右转一下（约 15°）",
     "grab": 8, "quota": 1},
]


class EnrollSession:
    """一次录入会话的状态机。

    调用方（UI/测试）逐帧喂入检测结果，会话内部完成：
    门控 → 去重 → 配额 → 采集即入库。
    """

    def __init__(self, student_no, name, dept=None,
                 recognizer=None, detector=None):
        self.student_no = student_no
        self.name = name
        self.dept = dept
        self.recognizer = recognizer
        self.detector = detector

        self.person_repo = PersonRepo()
        self.feature_repo = FeatureRepo()

        # 创建/复用 person：
        # - 学号不存在 → 新建人员
        # - 学号已存在但无人脸样本（或未录满）→ 复用该人员，继续采集
        # - 学号已录满 MAX_SAMPLES_PER_PERSON 张 → 拒绝
        existing = self.person_repo.get_by_student_no(student_no)
        if existing:
            cnt = self.feature_repo.count_by_person(existing["id"])
            if cnt >= config.MAX_SAMPLES_PER_PERSON:
                raise ValueError(
                    f"学号 {student_no} 已录满 {config.MAX_SAMPLES_PER_PERSON} 张样本")
            self.person_id = existing["id"]
            # 以数据库已有姓名/院系为准，避免同名异人、数据不一致
            self.name = existing["name"] or name
            self.dept = existing["dept"] or dept
        else:
            self.person_id = self.person_repo.create(student_no, name, dept)

        # 状态
        self.instruction_idx = 0            # 当前指令索引
        self.saved_in_instruction = 0       # 当前指令已保存数
        self.total_saved = 0                # 总已保存数
        self.last_status = ""               # 最近一次状态提示
        self.finished = False
        self.photos_dir = os.path.join(
            config.PHOTOS_DIR, str(student_no))
        os.makedirs(self.photos_dir, exist_ok=True)

    # ---------------- 主入口 ----------------
    def process_frame(self, img_bgr, faces, gate_result=None):
        """处理一帧。

        参数：
            img_bgr: 当前帧
            faces: YuNet 检测结果
            gate_result: 若已在外层做过门控可传入 (passed, reason)

        返回 dict，含状态信息，供 UI 展示。
        """
        if self.finished:
            return self._status("样本已采集完成（5/5）")

        # 门控
        if gate_result is None:
            from core.quality_gate import check_gate
            gate_result = check_gate(faces, img_bgr)
        passed, reason = gate_result
        if not passed:
            return self._status(reason)

        face = faces[0]

        # 提取特征
        aligned, feat = self.recognizer.align_and_feature(img_bgr, face)

        # 相似度去重
        existing = self.feature_repo.list_by_person(self.person_id)
        if self._is_duplicate(feat, existing):
            return self._status("该帧与已有样本重复，继续采集...")

        # 配额检查
        inst = INSTRUCTIONS[self.instruction_idx]
        if self.saved_in_instruction >= inst["quota"]:
            # 当前指令已满，切换到下一条
            if not self._next_instruction():
                self.finished = True
                return self._status("样本已采集完成（5/5）")
            inst = INSTRUCTIONS[self.instruction_idx]

        # 样本上限（双重保险）
        if self.total_saved >= config.MAX_SAMPLES_PER_PERSON:
            self.finished = True
            return self._status("样本已达上限 5 条")

        # 入库
        quality = float(face[14])
        self.feature_repo.add(self.person_id, feat, quality)
        self.saved_in_instruction += 1
        self.total_saved += 1

        # 保存原图 + 对齐图（相对路径）
        self._save_photos(img_bgr, aligned, face)

        # 判断是否需要切换指令
        if self.saved_in_instruction >= inst["quota"]:
            if self.total_saved >= config.MAX_SAMPLES_PER_PERSON:
                self.finished = True
                return self._status(f"已保存 {self.total_saved}/5（完成）")
            self._next_instruction()

        return self._status(f"已保存 {self.total_saved}/5")

    # ---------------- 内部 ----------------
    def _is_duplicate(self, feat, existing):
        """与已有样本余弦相似度 ≥ 0.92 判定重复。"""
        for e in existing:
            sim = float(np.dot(feat, e))  # 两者均已 L2 归一化
            if sim >= config.DUP_THRESHOLD:
                return True
        return False

    def _next_instruction(self):
        """切换到下一条指令。返回 False 表示已无下一条。"""
        if self.instruction_idx + 1 >= len(INSTRUCTIONS):
            return False
        self.instruction_idx += 1
        self.saved_in_instruction = 0
        return True

    def _save_photos(self, img_bgr, aligned, face):
        """保存原图 + 对齐图，相对路径。"""
        import cv2
        seq = self.total_saved  # 已保存序号（1 起）
        raw_name = f"{seq:02d}_raw.jpg"
        aligned_name = f"{seq:02d}_aligned.jpg"
        cv2.imwrite(os.path.join(self.photos_dir, raw_name), img_bgr)
        cv2.imwrite(os.path.join(self.photos_dir, aligned_name), aligned)
        # 记录到 face_photo 表
        conn = None
        from repository.db import get_conn
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO face_photo (person_id, raw_path, aligned_path) "
                "VALUES (?, ?, ?)",
                (self.person_id,
                 os.path.join(str(self.student_no), raw_name),
                 os.path.join(str(self.student_no), aligned_name)))
            conn.commit()
        finally:
            conn.close()

    def _status(self, msg):
        self.last_status = msg
        return {
            "student_no": self.student_no,
            "name": self.name,
            "instruction": INSTRUCTIONS[self.instruction_idx]["text"],
            "instruction_idx": self.instruction_idx,
            "saved_total": self.total_saved,
            "saved_in_instruction": self.saved_in_instruction,
            "finished": self.finished,
            "message": msg,
        }

    def current_instruction_text(self):
        return INSTRUCTIONS[self.instruction_idx]["text"]

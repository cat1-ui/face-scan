"""识别编排逻辑（service 层）。

核心职责（实施方案 M4）：
- 取帧 → 检测 → 门控 → 特征 → 1:N 匹配 → 判定 → 写库
- 时段去重（session_key = 日期 + AM/PM）
- 记录写入（含 4 诊断字段）

关键点：
- 拒识分支：低于阈值不强行匹配。
- gap < 0.08 时只提示、不自动写库（疑似张冠李戴）。
- 同一时段只记一条（session_key 唯一索引兜底）。
"""
import datetime

from core import config
from repository.attendance_repo import AttendanceRepo


def current_session_key(now=None):
    """生成时段 key：日期 + 上午/下午。"""
    now = now or datetime.datetime.now()
    period = "AM" if now.hour < 12 else "PM"
    return f"{now.date().isoformat()}-{period}"


class AttendService:
    def __init__(self, matcher):
        self.matcher = matcher
        self.attendance_repo = AttendanceRepo()

    def process(self, query_feature, sharpness=None, brightness=None):
        """对一帧的特征做识别，返回结果 dict。

        结果含：
            recognized / need_confirm / person_id / name / top1_sim / top2_sim
            action: 'matched'(已记录) | 'recorded'(刚记录) | 'confirm'(需人工确认)
                    | 'unknown'(未识别) | 'already'(该时段已记录)
        """
        m = self.matcher.match(query_feature)

        if not m["recognized"]:
            return {
                "recognized": False,
                "need_confirm": False,
                "person_id": None,
                "name": None,
                "top1_sim": m["top1_sim"],
                "top2_sim": m["top2_sim"],
                "action": "unknown",
            }

        person_id = m["top1_person_id"]
        name = self.matcher.person_name(person_id)

        # 相似脸：只提示，不写库
        if m["need_confirm"]:
            return {
                "recognized": True,
                "need_confirm": True,
                "person_id": person_id,
                "name": name,
                "top1_sim": m["top1_sim"],
                "top2_sim": m["top2_sim"],
                "action": "confirm",
            }

        # 时段去重
        session_key = current_session_key()
        if self.attendance_repo.exists_session(person_id, session_key):
            return {
                "recognized": True,
                "need_confirm": False,
                "person_id": person_id,
                "name": name,
                "top1_sim": m["top1_sim"],
                "top2_sim": m["top2_sim"],
                "action": "already",
                "session_key": session_key,
            }

        # 写库（4 诊断字段）
        self.attendance_repo.add(
            person_id, session_key,
            similarity=m["top1_sim"],
            runner_up_sim=m["top2_sim"],
            sharpness=sharpness,
            brightness=brightness)

        return {
            "recognized": True,
            "need_confirm": False,
            "person_id": person_id,
            "name": name,
            "top1_sim": m["top1_sim"],
            "top2_sim": m["top2_sim"],
            "action": "recorded",
            "session_key": session_key,
        }

"""attendance 表数据访问。"""
import datetime

from repository.db import get_conn


class AttendanceRepo:
    def add(self, person_id, session_key, similarity=None,
            runner_up_sim=None, sharpness=None, brightness=None,
            method="face"):
        """写入一条考勤记录（含 4 个诊断字段）。"""
        now = datetime.datetime.now().isoformat(timespec="seconds")
        conn = get_conn()
        try:
            cur = conn.execute(
                "INSERT INTO attendance (person_id, ts, session_key, method, "
                "similarity, runner_up_sim, sharpness, brightness) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (person_id, now, session_key, method, similarity,
                 runner_up_sim, sharpness, brightness))
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def exists_session(self, person_id, session_key):
        """判断某人在某时段是否已打卡。"""
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT id FROM attendance WHERE person_id = ? AND "
                "session_key = ?", (person_id, session_key)).fetchone()
            return row is not None
        finally:
            conn.close()

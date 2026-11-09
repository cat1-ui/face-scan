"""person 表数据访问。"""
import datetime

from repository.db import get_conn


class PersonRepo:
    def create(self, student_no, name, dept=None):
        """创建 person 记录。返回新记录的 id。

        若学号已存在，抛出 ValueError。
        """
        conn = get_conn()
        try:
            exists = conn.execute(
                "SELECT id FROM person WHERE student_no = ?",
                (student_no,)).fetchone()
            if exists:
                raise ValueError(f"学号 {student_no} 已存在，请勿重复录入")
            now = datetime.datetime.now().isoformat(timespec="seconds")
            cur = conn.execute(
                "INSERT INTO person (student_no, name, dept, face_status, "
                "created_at) VALUES (?, ?, ?, 'active', ?)",
                (student_no, name, dept, now))
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_by_student_no(self, student_no):
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM person WHERE student_no = ?",
                (student_no,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get(self, person_id):
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM person WHERE id = ?",
                (person_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def set_face_status(self, person_id, status):
        conn = get_conn()
        try:
            conn.execute(
                "UPDATE person SET face_status = ? WHERE id = ?",
                (status, person_id))
            conn.commit()
        finally:
            conn.close()

    def list_all(self):
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM person ORDER BY id").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

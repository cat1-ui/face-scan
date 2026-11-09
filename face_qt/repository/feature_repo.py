"""face_feature 表数据访问（含 BLOB ↔ numpy 序列化）。

关键点（A3-4）：BLOB 存取必须 dtype 一致（float32），
用 tobytes() 存、frombuffer(dtype=float32) 取，保证 array_equal。
"""
import numpy as np

from repository.db import get_conn


def feature_to_blob(feat):
    """numpy 特征 → BLOB 字节。feat 为 float32 数组。"""
    return np.asarray(feat, dtype=np.float32).tobytes()


def blob_to_feature(blob):
    """BLOB 字节 → numpy float32 特征。"""
    return np.frombuffer(blob, dtype=np.float32)


class FeatureRepo:
    def add(self, person_id, feature, quality=None):
        """新增一条特征。feature 为 128 维 float32 数组。"""
        blob = feature_to_blob(feature)
        conn = get_conn()
        try:
            cur = conn.execute(
                "INSERT INTO face_feature (person_id, feature, quality) "
                "VALUES (?, ?, ?)",
                (person_id, blob, quality))
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def count_by_person(self, person_id):
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM face_feature WHERE person_id = ?",
                (person_id,)).fetchone()
            return row["c"]
        finally:
            conn.close()

    def list_by_person(self, person_id):
        """返回该人的所有特征（numpy 数组列表）。"""
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM face_feature WHERE person_id = ? "
                "ORDER BY id", (person_id,)).fetchall()
            return [blob_to_feature(r["feature"]) for r in rows]
        finally:
            conn.close()

    def list_all_active(self):
        """返回所有 active 状态人员的 (person_id, feature) 列表。

        供 M4 gallery 加载使用。
        """
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT f.person_id, f.feature FROM face_feature f "
                "JOIN person p ON p.id = f.person_id "
                "WHERE p.face_status = 'active' "
                "ORDER BY f.id").fetchall()
            return [(r["person_id"], blob_to_feature(r["feature"]))
                    for r in rows]
        finally:
            conn.close()

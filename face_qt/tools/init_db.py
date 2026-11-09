"""初始化数据库：建库 + 建表 + PRAGMA 并发设置。

严格按《人脸识别系统_设计方案》7.2 节手写 SQL，不使用 ORM。
运行：.venv/Scripts/python.exe tools/init_db.py
"""
import os
import sqlite3

# 数据库位于跨项目共享目录 face_data/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # face_qt/
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "face_data")          # face_data/
DB_PATH = os.path.join(DATA_DIR, "attendance.db")


def get_conn():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # 并发设置（三端共享一个库，必做）
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


SCHEMA_SQL = """
-- ============================================================
-- 表
-- ============================================================
CREATE TABLE IF NOT EXISTS person (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_no  TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL,
    dept        TEXT,
    face_status TEXT    NOT NULL DEFAULT 'active',   -- active / disabled
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS face_feature (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER NOT NULL,
    feature    BLOB    NOT NULL,                    -- 128 维 float32
    quality    REAL,                                -- 检测置信度
    created_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS face_photo (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id    INTEGER NOT NULL,
    raw_path     TEXT,                               -- 相对路径
    aligned_path TEXT,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS attendance (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     INTEGER NOT NULL,
    ts            TEXT    NOT NULL,                  -- ISO8601
    session_key   TEXT    NOT NULL,                  -- 2026-10-12-AM
    method        TEXT    NOT NULL DEFAULT 'face',
    similarity    REAL,
    runner_up_sim REAL,                              -- 诊断用
    sharpness     REAL,                              -- 诊断用
    brightness    REAL,                              -- 诊断用
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE CASCADE
);

-- ============================================================
-- 索引
-- ============================================================
CREATE UNIQUE INDEX IF NOT EXISTS idx_att_unique
    ON attendance(person_id, session_key);           -- 同一时段只记一条
CREATE INDEX IF NOT EXISTS idx_feature_person ON face_feature(person_id);
CREATE INDEX IF NOT EXISTS idx_attendance_ts  ON attendance(ts);
CREATE INDEX IF NOT EXISTS idx_att_session    ON attendance(session_key);
CREATE INDEX IF NOT EXISTS idx_person_no      ON person(student_no);
"""


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA_SQL)
    conn.commit()

    # 校验
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    conn.close()

    print("数据库路径:", DB_PATH)
    print("已建表:", tables)
    print("journal_mode:", journal)


if __name__ == "__main__":
    init_db()

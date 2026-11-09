"""数据库连接与 PRAGMA 设置。

严格按设计方案 7.2 节：WAL + busy_timeout + foreign_keys。
"""
import os
import sqlite3

from core import config


def get_conn():
    """获取数据库连接，已配置并发 PRAGMA。"""
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

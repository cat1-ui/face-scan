"""人脸考勤系统 · Web 后端（FastAPI）。

读取 QT 端写入的共享数据库 face_data/attendance.db，
提供考勤查询、人脸样本查看、Excel 导出等接口。

运行：.venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000
"""
import os
import sqlite3
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# 数据库路径：跨项目共享（QT 端也写这个库）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "face_data")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
PHOTOS_DIR = os.path.join(DATA_DIR, "photos")

app = FastAPI(title="人脸考勤系统 API", version="1.0.0")

# 允许前端跨域访问（Vue dev server 默认 5173 端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


# ---------------- 健康检查 ----------------
@app.get("/api/health")
def health():
    return {"status": "ok", "db": os.path.exists(DB_PATH)}


# ---------------- 学生列表（含搜索） ----------------
class PersonCreate(BaseModel):
    student_no: str = Field(..., min_length=1, description="学号")
    name: str = Field(..., min_length=1, description="姓名")
    dept: str | None = None


class PersonUpdate(BaseModel):
    name: str = Field(..., min_length=1, description="姓名")
    dept: str | None = None


@app.get("/api/persons")
def list_persons(
    keyword: Optional[str] = Query(None, description="按学号/姓名模糊搜索"),
):
    conn = get_conn()
    try:
        sql = (
            "SELECT p.id, p.student_no, p.name, p.dept, p.face_status, "
            "p.created_at, "
            "(SELECT COUNT(*) FROM face_feature f WHERE f.person_id=p.id) "
            "AS sample_count, "
            "(SELECT COUNT(*) FROM face_photo ph WHERE ph.person_id=p.id) "
            "AS photo_count "
            "FROM person p"
        )
        params = []
        if keyword:
            sql += " WHERE p.student_no LIKE ? OR p.name LIKE ?"
            like = f"%{keyword}%"
            params += [like, like]
        sql += " ORDER BY p.id"
        rows = conn.execute(sql, params).fetchall()
        return {"data": [dict(r) for r in rows]}
    finally:
        conn.close()


# ---------------- 新增人员 ----------------
@app.post("/api/persons", status_code=201)
def create_person(payload: PersonCreate):
    conn = get_conn()
    try:
        # 学号唯一性检查
        exists = conn.execute(
            "SELECT id FROM person WHERE student_no = ?",
            (payload.student_no,)).fetchone()
        if exists:
            raise HTTPException(status_code=409, detail="学号已存在")
        now = datetime.now().isoformat(timespec="seconds")
        cur = conn.execute(
            "INSERT INTO person (student_no, name, dept, face_status, "
            "created_at) VALUES (?, ?, ?, 'active', ?)",
            (payload.student_no, payload.name, payload.dept, now))
        conn.commit()
        return {"id": cur.lastrowid, "message": "新增成功"}
    finally:
        conn.close()


# ---------------- 修改人员（姓名/院系） ----------------
@app.put("/api/persons/{person_id}")
def update_person(person_id: int, payload: PersonUpdate):
    conn = get_conn()
    try:
        exists = conn.execute(
            "SELECT id FROM person WHERE id = ?", (person_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="人员不存在")
        conn.execute(
            "UPDATE person SET name = ?, dept = ? WHERE id = ?",
            (payload.name, payload.dept, person_id))
        conn.commit()
        return {"message": "修改成功"}
    finally:
        conn.close()


# ---------------- 删除人员（级联删除特征/照片/考勤） ----------------
@app.delete("/api/persons/{person_id}")
def delete_person(person_id: int):
    conn = get_conn()
    try:
        exists = conn.execute(
            "SELECT id FROM person WHERE id = ?", (person_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="人员不存在")
        # 外键 ON DELETE CASCADE 自动连带删除 face_feature/face_photo/attendance
        conn.execute("DELETE FROM person WHERE id = ?", (person_id,))
        conn.commit()
        return {"message": "删除成功（已级联删除特征/照片/考勤）"}
    finally:
        conn.close()


# ---------------- 考勤记录 ----------------
@app.get("/api/attendance")
def list_attendance(
    date: Optional[str] = Query(None, description="日期 YYYY-MM-DD"),
    student_no: Optional[str] = Query(None),
):
    conn = get_conn()
    try:
        sql = (
            "SELECT a.id, p.student_no, p.name, a.ts, a.session_key, "
            "a.method, a.similarity, a.runner_up_sim, a.sharpness, "
            "a.brightness "
            "FROM attendance a JOIN person p ON p.id = a.person_id "
            "WHERE 1=1"
        )
        params = []
        if date:
            sql += " AND a.ts LIKE ?"
            params.append(f"{date}%")
        if student_no:
            sql += " AND p.student_no = ?"
            params.append(student_no)
        sql += " ORDER BY a.ts DESC"
        rows = conn.execute(sql, params).fetchall()
        return {"data": [dict(r) for r in rows]}
    finally:
        conn.close()


# ---------------- 人脸样本 ----------------
@app.get("/api/persons/{person_id}/photos")
def list_photos(person_id: int):
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, raw_path, aligned_path FROM face_photo "
            "WHERE person_id = ? ORDER BY id", (person_id,)).fetchall()
        return {"data": [dict(r) for r in rows]}
    finally:
        conn.close()


# 人脸样本图片（对齐图）
@app.get("/api/photos/{path:path}")
def get_photo(path: str):
    full = os.path.join(PHOTOS_DIR, path)
    if not os.path.exists(full):
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(full)


# ---------------- 删除单张人脸样本（照片 + 对应特征 + 物理文件） ----------------
@app.delete("/api/persons/{person_id}/photos/{photo_id}")
def delete_photo(person_id: int, photo_id: int):
    """删除指定照片，并同步删除对应的 face_feature 与物理文件。

    照片与特征按 id 升序一一对应（录入时同步各插入一条），
    故删除第 N 张照片时，删除同序号的第 N 条特征，保持二者数量一致。
    """
    conn = get_conn()
    try:
        # 定位该照片
        photo = conn.execute(
            "SELECT id, raw_path, aligned_path FROM face_photo "
            "WHERE id = ? AND person_id = ?", (photo_id, person_id)).fetchone()
        if not photo:
            raise HTTPException(status_code=404, detail="照片不存在")

        # 该照片在本人照片中的序号（第几张）
        seq = conn.execute(
            "SELECT COUNT(*) FROM face_photo WHERE person_id = ? AND id < ?",
            (person_id, photo_id)).fetchone()[0]

        # 删除对应的第 seq 条特征（按 id 升序）
        feat_rows = conn.execute(
            "SELECT id FROM face_feature WHERE person_id = ? ORDER BY id",
            (person_id,)).fetchall()
        if seq < len(feat_rows):
            conn.execute(
                "DELETE FROM face_feature WHERE id = ?",
                (feat_rows[seq]["id"],))

        # 删除照片记录
        conn.execute("DELETE FROM face_photo WHERE id = ?", (photo_id,))
        conn.commit()

        # 删除物理文件（不存在的路径忽略）
        for rel in (photo["raw_path"], photo["aligned_path"]):
            if rel:
                full = os.path.join(PHOTOS_DIR, rel)
                if os.path.exists(full):
                    os.remove(full)

        return {"message": "已删除该样本（照片 + 特征）"}
    finally:
        conn.close()


# ---------------- 统计 ----------------
@app.get("/api/stats")
def stats():
    conn = get_conn()
    try:
        total_persons = conn.execute(
            "SELECT COUNT(*) FROM person").fetchone()[0]
        total_attendance = conn.execute(
            "SELECT COUNT(*) FROM attendance").fetchone()[0]
        # 按日期统计
        by_date = conn.execute(
            "SELECT substr(ts,1,10) AS d, COUNT(*) AS c "
            "FROM attendance GROUP BY d ORDER BY d").fetchall()
        return {
            "total_persons": total_persons,
            "total_attendance": total_attendance,
            "by_date": [dict(r) for r in by_date],
        }
    finally:
        conn.close()


# ---------------- Excel 导出 ----------------
@app.get("/api/attendance/export")
def export_attendance(
    date: Optional[str] = Query(None),
):
    from openpyxl import Workbook
    conn = get_conn()
    try:
        sql = (
            "SELECT p.student_no, p.name, a.ts, a.session_key, a.similarity, "
            "a.runner_up_sim, a.sharpness, a.brightness "
            "FROM attendance a JOIN person p ON p.id = a.person_id "
            "WHERE 1=1"
        )
        params = []
        if date:
            sql += " AND a.ts LIKE ?"
            params.append(f"{date}%")
        sql += " ORDER BY a.ts DESC"
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    wb = Workbook()
    ws = wb.active
    ws.title = "考勤记录"
    headers = ["学号", "姓名", "时间", "时段", "相似度", "次高相似度",
               "清晰度", "亮度"]
    ws.append(headers)
    for r in rows:
        ws.append([r["student_no"], r["name"], r["ts"], r["session_key"],
                   round(r["similarity"], 4) if r["similarity"] else None,
                   round(r["runner_up_sim"], 4) if r["runner_up_sim"] else None,
                   round(r["sharpness"], 2) if r["sharpness"] else None,
                   round(r["brightness"], 2) if r["brightness"] else None])

    out_dir = os.path.join(BASE_DIR, "exports")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"attendance_{ts}.xlsx")
    wb.save(path)
    return FileResponse(
        path, filename=f"考勤记录_{ts}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument"
                   ".spreadsheetml.sheet")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

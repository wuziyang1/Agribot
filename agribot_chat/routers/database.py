import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pymysql
from pymongo import MongoClient, DESCENDING
from pymongo.database import Database


# ---------------------------------------------------------------------------
#  MySQL 连接配置（用户表）
# ---------------------------------------------------------------------------

def _get_mysql_conn():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "agribot"),
        password=os.getenv("MYSQL_PASSWORD", "agribot123"),
        database=os.getenv("MYSQL_DATABASE", "agribot"),
        charset=os.getenv("MYSQL_CHARSET", "utf8mb4"),
        cursorclass=pymysql.cursors.DictCursor,
    )


# ---------------------------------------------------------------------------
#  MongoDB 连接配置（会话 & 消息）
# ---------------------------------------------------------------------------

def _load_mongo_uri() -> str:
    host = os.getenv("MONGO_HOST", "127.0.0.1")
    port = os.getenv("MONGO_PORT", "27017")
    user = os.getenv("MONGO_USER", "agribot")
    password = os.getenv("MONGO_PASSWORD", "agribot123")
    database = os.getenv("MONGO_DATABASE", "agribot")
    return f"mongodb://{user}:{password}@{host}:{port}/{database}?authSource=admin"


def _get_db_name() -> str:
    return os.getenv("MONGO_DATABASE", "agribot")


_client: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(_load_mongo_uri(), serverSelectionTimeoutMS=5000)
    return _client


def _db() -> Database:
    return _get_client()[_get_db_name()]


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
#  初始化：确保表/集合和索引存在
# ---------------------------------------------------------------------------

def ensure_users_table_exists(**_: Any) -> None:
    """MySQL users 表由建表 SQL 管理，此处仅做连通性检查。"""
    conn = _get_mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users LIMIT 1")
    finally:
        conn.close()


def ensure_sessions_table_exists(**_: Any) -> None:
    db = _db()
    coll = db["sessions"]
    coll.create_index("session_id", unique=True)
    coll.create_index([("user_id", 1), ("updated_at", DESCENDING)])
    coll.create_index([("user_id", 1), ("is_active", 1)])


def ensure_messages_table_exists(**_: Any) -> None:
    db = _db()
    coll = db["messages"]
    coll.create_index([("session_id", 1), ("created_at", 1)])
    coll.create_index("created_at")


# ---------------------------------------------------------------------------
#  用户相关（MySQL）
# ---------------------------------------------------------------------------

def _row_to_user(row: Optional[dict]) -> Optional[Dict[str, Any]]:
    """把 MySQL 行转换成统一的用户字典。"""
    if row is None:
        return None
    return {
        "id": row.get("id"),
        "username": row.get("username"),
        "email": row.get("email"),
        "hashed_password": row.get("hashed_password"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def get_user_by_username(username: str, **_: Any) -> Optional[Dict[str, Any]]:
    conn = _get_mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            return _row_to_user(cur.fetchone())
    finally:
        conn.close()


def get_user_by_email(email: str, **_: Any) -> Optional[Dict[str, Any]]:
    conn = _get_mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            return _row_to_user(cur.fetchone())
    finally:
        conn.close()


def get_user_by_username_and_email(
    username: str,
    email: str,
    **_: Any,
) -> Optional[Dict[str, Any]]:
    conn = _get_mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM users WHERE username = %s AND email = %s",
                (username, email),
            )
            return _row_to_user(cur.fetchone())
    finally:
        conn.close()


def create_user(
    *,
    username: str,
    email: Optional[str],
    hashed_password: str,
    **_: Any,
) -> int:
    conn = _get_mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, email, hashed_password) VALUES (%s, %s, %s)",
                (username, email, hashed_password),
            )
            conn.commit()
            return cur.lastrowid
    except pymysql.IntegrityError as e:
        conn.rollback()
        raise ValueError("用户名已存在") from e
    finally:
        conn.close()


def update_user_password_by_username(
    *,
    username: str,
    hashed_password: str,
    **_: Any,
) -> bool:
    conn = _get_mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET hashed_password = %s WHERE username = %s",
                (hashed_password, username),
            )
            conn.commit()
            return cur.rowcount > 0
    finally:
        conn.close()


def update_user_password(username: str, hashed_password: str, **_: Any) -> bool:
    return update_user_password_by_username(
        username=username, hashed_password=hashed_password,
    )


def update_user_profile(
    *,
    user_id: int,
    username: Optional[str] = None,
    email: Optional[str] = None,
    **_: Any,
) -> Tuple[bool, str]:
    if username is not None:
        username = (username or "").strip()
        if not username:
            return False, "用户名不能为空"
        if len(username) > 50:
            return False, "用户名过长"
        existing = get_user_by_username(username)
        if existing and existing.get("id") != user_id:
            return False, "用户名已被占用"

    if email is not None:
        email = (email or "").strip()
        if email and len(email) > 100:
            return False, "邮箱过长"
        if email:
            existing = get_user_by_email(email)
            if existing and existing.get("id") != user_id:
                return False, "邮箱已被占用"

    sets = []
    params = []
    if username is not None:
        sets.append("username = %s")
        params.append(username)
    if email is not None:
        sets.append("email = %s")
        params.append(email or None)
    if not sets:
        return True, ""

    params.append(user_id)
    conn = _get_mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE users SET {', '.join(sets)} WHERE id = %s",
                tuple(params),
            )
            conn.commit()
            return cur.rowcount > 0, ""
    except pymysql.IntegrityError:
        conn.rollback()
        return False, "用户名或邮箱已被占用"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
#  会话相关（MongoDB）
# ---------------------------------------------------------------------------

def create_session(*, user_id: int, **_: Any) -> Dict[str, Any]:
    sess_id = str(uuid.uuid4())
    now = _now()
    db = _db()
    db["sessions"].update_many(
        {"user_id": user_id},
        {"$set": {"is_active": 0}},
    )
    db["sessions"].insert_one({
        "session_id": sess_id,
        "user_id": user_id,
        "title": None,
        "is_active": 1,
        "created_at": now,
        "updated_at": now,
    })
    return {"session_id": sess_id, "title": None, "is_active": 1}


def list_sessions(*, user_id: int, **_: Any) -> List[Dict[str, Any]]:
    cursor = (
        _db()["sessions"]
        .find(
            {"user_id": user_id},
            {"_id": 0, "session_id": 1, "title": 1, "is_active": 1,
             "created_at": 1, "updated_at": 1},
        )
        .sort([("updated_at", DESCENDING), ("created_at", DESCENDING)])
    )
    return list(cursor)


def set_active_session(*, user_id: int, session_id: str, **_: Any) -> bool:
    db = _db()
    db["sessions"].update_many(
        {"user_id": user_id},
        {"$set": {"is_active": 0}},
    )
    result = db["sessions"].update_one(
        {"user_id": user_id, "session_id": session_id},
        {"$set": {"is_active": 1}},
    )
    return result.modified_count > 0


def update_session_title(
    *,
    user_id: int,
    session_id: str,
    title: Optional[str],
    **_: Any,
) -> bool:
    title = (title or "").strip() or None
    if title and len(title) > 200:
        title = title[:200]
    result = _db()["sessions"].update_one(
        {"user_id": user_id, "session_id": session_id},
        {"$set": {"title": title}},
    )
    return result.matched_count > 0


def delete_session(*, user_id: int, session_id: str, **_: Any) -> bool:
    db = _db()
    result = db["sessions"].delete_one(
        {"user_id": user_id, "session_id": session_id},
    )
    if result.deleted_count > 0:
        db["messages"].delete_many({"session_id": session_id})
        return True
    return False


def get_session_by_id(*, user_id: int, session_id: str, **_: Any) -> Optional[Dict[str, Any]]:
    doc = _db()["sessions"].find_one(
        {"user_id": user_id, "session_id": session_id},
        {"_id": 0, "session_id": 1, "title": 1, "is_active": 1,
         "created_at": 1, "updated_at": 1},
    )
    return doc


# ---------------------------------------------------------------------------
#  消息相关（MongoDB）
# ---------------------------------------------------------------------------

def list_messages(*, user_id: int, session_id: str, **_: Any) -> List[Dict[str, Any]]:
    db = _db()
    sess = db["sessions"].find_one(
        {"user_id": user_id, "session_id": session_id},
        {"_id": 1},
    )
    if not sess:
        return []
    cursor = (
        db["messages"]
        .find(
            {"session_id": session_id},
            {"_id": 0, "role": 1, "content": 1, "created_at": 1},
        )
        .sort("created_at", 1)
    )
    return list(cursor)


def create_message(
    *,
    user_id: int,
    session_id: str,
    role: str,
    content: str,
    **_: Any,
) -> bool:
    role = (role or "").strip()
    if role not in {"user", "assistant", "system"}:
        raise ValueError("invalid role")
    content = content or ""
    if not content.strip():
        return False

    db = _db()
    sess = db["sessions"].find_one(
        {"user_id": user_id, "session_id": session_id},
        {"_id": 1},
    )
    if not sess:
        return False

    now = _now()
    db["messages"].insert_one({
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": now,
    })
    db["sessions"].update_one(
        {"user_id": user_id, "session_id": session_id},
        {"$set": {"updated_at": now}},
    )
    return True

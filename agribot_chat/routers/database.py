import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pymysql
from pymysql.cursors import DictCursor


@dataclass(frozen=True)
class MySQLConfig:
    host: str
    user: str
    password: str
    database: str
    port: int
    charset: str


def load_mysql_config() -> MySQLConfig:
    """
    从环境变量读取 MySQL 配置（默认本机 MySQL）。

    环境变量：
    - MYSQL_HOST / MYSQL_PORT
    - MYSQL_USER / MYSQL_PASSWORD
    - MYSQL_DATABASE
    - MYSQL_CHARSET（默认 utf8mb4）
    """
    return MySQLConfig(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "oceanhub"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        charset=os.getenv("MYSQL_CHARSET", "utf8mb4"),
    )


@contextmanager
def mysql_conn(cfg: Optional[MySQLConfig] = None) -> Iterator[pymysql.connections.Connection]:
    cfg = cfg or load_mysql_config()
    conn = pymysql.connect(
        host=cfg.host,
        user=cfg.user,
        password=cfg.password,
        database=cfg.database,
        port=cfg.port,
        charset=cfg.charset,
        cursorclass=DictCursor,
        autocommit=True,
        read_timeout=10,
        write_timeout=10,
        connect_timeout=5,
    )
    try:
        yield conn
    finally:
        conn.close()


def ensure_users_table_exists(cfg: Optional[MySQLConfig] = None) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS `users` (
      `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '用户ID',
      `username` VARCHAR(50) NOT NULL COMMENT '用户名',
      `email` VARCHAR(100) DEFAULT NULL COMMENT '邮箱地址',
      `hashed_password` VARCHAR(255) NOT NULL COMMENT '密码哈希值',
      `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
      `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
      PRIMARY KEY (`id`),
      UNIQUE KEY `uk_username` (`username`),
      KEY `idx_email` (`email`),
      KEY `idx_created_at` (`created_at`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)


def ensure_sessions_table_exists(cfg: Optional[MySQLConfig] = None) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS `sessions` (
        `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '会话ID（自增）',
        `session_id` VARCHAR(36) NOT NULL COMMENT '会话UUID，唯一标识',
        `user_id` BIGINT UNSIGNED NOT NULL COMMENT '用户ID',
        `title` VARCHAR(200) DEFAULT NULL COMMENT '会话标题',
        `is_active` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否为当前活跃会话：0=否，1=是',
        `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
        `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
        PRIMARY KEY (`id`),
        UNIQUE KEY `uk_session_id` (`session_id`),
        KEY `idx_user_id` (`user_id`),
        KEY `idx_is_active` (`is_active`),
        KEY `idx_created_at` (`created_at`),
        KEY `idx_updated_at` (`updated_at`),
        CONSTRAINT `fk_sessions_user_id` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='会话表';
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)


def ensure_messages_table_exists(cfg: Optional[MySQLConfig] = None) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS `messages` (
        `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '消息ID',
        `session_id` VARCHAR(36) NOT NULL COMMENT '会话UUID',
        `role` VARCHAR(20) NOT NULL COMMENT '消息角色：user=用户，assistant=AI助手，system=系统',
        `content` TEXT NOT NULL COMMENT '消息内容',
        `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
        PRIMARY KEY (`id`),
        KEY `idx_session_id` (`session_id`),
        KEY `idx_role` (`role`),
        KEY `idx_created_at` (`created_at`),
        FULLTEXT KEY `ft_content` (`content`),
        CONSTRAINT `fk_messages_session_id` FOREIGN KEY (`session_id`) REFERENCES `sessions` (`session_id`) ON DELETE CASCADE ON UPDATE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='消息表';
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)


def get_user_by_username(username: str, *, cfg: Optional[MySQLConfig] = None) -> Optional[Dict[str, Any]]:
    sql = """
    SELECT id, username, email, hashed_password, created_at, updated_at
    FROM users
    WHERE username=%s
    LIMIT 1
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (username,))
            return cur.fetchone()


def get_user_by_email(email: str, *, cfg: Optional[MySQLConfig] = None) -> Optional[Dict[str, Any]]:
    sql = """
    SELECT id, username, email, hashed_password, created_at, updated_at
    FROM users
    WHERE email=%s
    LIMIT 1
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (email,))
            return cur.fetchone()


def get_user_by_username_and_email(
    username: str,
    email: str,
    *,
    cfg: Optional[MySQLConfig] = None,
) -> Optional[Dict[str, Any]]:
    sql = """
    SELECT id, username, email, hashed_password, created_at, updated_at
    FROM users
    WHERE username=%s AND email=%s
    LIMIT 1
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (username, email))
            return cur.fetchone()


def create_user(
    *,
    username: str,
    email: Optional[str],
    hashed_password: str,
    cfg: Optional[MySQLConfig] = None,
) -> int:
    sql = """
    INSERT INTO users (username, email, hashed_password)
    VALUES (%s, %s, %s)
    """
    try:
        with mysql_conn(cfg) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (username, email, hashed_password))
                return int(cur.lastrowid)
    except pymysql.err.IntegrityError as e:
        raise ValueError("用户名已存在") from e


def update_user_password_by_username(
    *,
    username: str,
    hashed_password: str,
    cfg: Optional[MySQLConfig] = None,
) -> bool:
    sql = """
    UPDATE users
    SET hashed_password=%s
    WHERE username=%s
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (hashed_password, username))
            return cur.rowcount > 0


def update_user_password(username: str, hashed_password: str, *, cfg: Optional[MySQLConfig] = None) -> bool:
    return update_user_password_by_username(username=username, hashed_password=hashed_password, cfg=cfg)


def update_user_profile(
    *,
    user_id: int,
    username: Optional[str] = None,
    email: Optional[str] = None,
    cfg: Optional[MySQLConfig] = None,
) -> Tuple[bool, str]:
    """
    更新用户资料（用户名、邮箱）。返回 (成功, 错误信息)。
    """
    if username is not None:
        username = (username or "").strip()
        if not username:
            return False, "用户名不能为空"
        if len(username) > 50:
            return False, "用户名过长"
        # 检查用户名是否被其他用户占用
        existing = get_user_by_username(username, cfg=cfg)
        if existing and existing.get("id") != user_id:
            return False, "用户名已被占用"

    if email is not None:
        email = (email or "").strip()
        if email and len(email) > 100:
            return False, "邮箱过长"
        if email:
            existing = get_user_by_email(email, cfg=cfg)
            if existing and existing.get("id") != user_id:
                return False, "邮箱已被占用"

    updates = []
    params = []
    if username is not None:
        updates.append("username=%s")
        params.append(username)
    if email is not None:
        updates.append("email=%s")
        params.append(email or None)
    if not updates:
        return True, ""

    params.append(user_id)
    sql = f"UPDATE users SET {', '.join(updates)} WHERE id=%s"
    try:
        with mysql_conn(cfg) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.rowcount > 0, ""
    except pymysql.err.IntegrityError:
        return False, "用户名或邮箱已被占用"


def create_session(*, user_id: int, cfg: Optional[MySQLConfig] = None) -> Dict[str, Any]:
    """新建会话，并将其设为活跃会话。"""
    sess_id = str(uuid.uuid4())
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            # 取消该用户的其他活跃会话
            cur.execute("UPDATE sessions SET is_active=0 WHERE user_id=%s", (user_id,))
            cur.execute(
                "INSERT INTO sessions (session_id, user_id, title, is_active) VALUES (%s, %s, %s, 1)",
                (sess_id, user_id, None),
            )
    return {"session_id": sess_id, "title": None, "is_active": 1}


def list_sessions(*, user_id: int, cfg: Optional[MySQLConfig] = None) -> List[Dict[str, Any]]:
    sql = """
    SELECT session_id, title, is_active, created_at, updated_at
    FROM sessions
    WHERE user_id=%s
    ORDER BY updated_at DESC, created_at DESC
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            return list(cur.fetchall() or [])


def set_active_session(*, user_id: int, session_id: str, cfg: Optional[MySQLConfig] = None) -> bool:
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE sessions SET is_active=0 WHERE user_id=%s", (user_id,))
            cur.execute(
                "UPDATE sessions SET is_active=1 WHERE user_id=%s AND session_id=%s",
                (user_id, session_id),
            )
            return cur.rowcount > 0


def update_session_title(
    *,
    user_id: int,
    session_id: str,
    title: Optional[str],
    cfg: Optional[MySQLConfig] = None,
) -> bool:
    title = (title or "").strip() or None
    if title and len(title) > 200:
        title = title[:200]
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET title=%s WHERE user_id=%s AND session_id=%s",
                (title, user_id, session_id),
            )
            return cur.rowcount > 0


def delete_session(*, user_id: int, session_id: str, cfg: Optional[MySQLConfig] = None) -> bool:
    """直接删除会话（同时会因外键 CASCADE 删除该会话下所有消息）。"""
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM sessions WHERE user_id=%s AND session_id=%s",
                (user_id, session_id),
            )
            return cur.rowcount > 0


def get_session_by_id(*, user_id: int, session_id: str, cfg: Optional[MySQLConfig] = None) -> Optional[Dict[str, Any]]:
    sql = """
    SELECT session_id, title, is_active, created_at, updated_at
    FROM sessions
    WHERE user_id=%s AND session_id=%s
    LIMIT 1
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, session_id))
            return cur.fetchone()


def list_messages(*, user_id: int, session_id: str, cfg: Optional[MySQLConfig] = None) -> List[Dict[str, Any]]:
    sql = """
    SELECT m.id, m.role, m.content, m.created_at
    FROM messages m
    JOIN sessions s ON s.session_id=m.session_id
    WHERE s.user_id=%s AND s.session_id=%s
    ORDER BY m.id ASC
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, session_id))
            return list(cur.fetchall() or [])


def create_message(
    *,
    user_id: int,
    session_id: str,
    role: str,
    content: str,
    cfg: Optional[MySQLConfig] = None,
) -> bool:
    role = (role or "").strip()
    if role not in {"user", "assistant", "system"}:
        raise ValueError("invalid role")
    content = content or ""
    if not content.strip():
        return False
    sql = """
    INSERT INTO messages (session_id, role, content)
    SELECT s.session_id, %s, %s
    FROM sessions s
    WHERE s.user_id=%s AND s.session_id=%s
    LIMIT 1
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (role, content, user_id, session_id))
            inserted = cur.rowcount > 0
            cur.execute(
                "UPDATE sessions SET updated_at=NOW() WHERE user_id=%s AND session_id=%s",
                (user_id, session_id),
            )
            return inserted


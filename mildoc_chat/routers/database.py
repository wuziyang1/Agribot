import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

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
      `deleted_at` DATETIME DEFAULT NULL COMMENT '删除时间（软删除）',
      PRIMARY KEY (`id`),
      UNIQUE KEY `uk_username` (`username`),
      KEY `idx_email` (`email`),
      KEY `idx_deleted_at` (`deleted_at`),
      KEY `idx_created_at` (`created_at`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)


def get_user_by_username(username: str, *, cfg: Optional[MySQLConfig] = None) -> Optional[Dict[str, Any]]:
    sql = """
    SELECT id, username, email, hashed_password, created_at, updated_at, deleted_at
    FROM users
    WHERE username=%s AND deleted_at IS NULL
    LIMIT 1
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (username,))
            return cur.fetchone()


def get_user_by_email(email: str, *, cfg: Optional[MySQLConfig] = None) -> Optional[Dict[str, Any]]:
    sql = """
    SELECT id, username, email, hashed_password, created_at, updated_at, deleted_at
    FROM users
    WHERE email=%s AND deleted_at IS NULL
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
    SELECT id, username, email, hashed_password, created_at, updated_at, deleted_at
    FROM users
    WHERE username=%s AND email=%s AND deleted_at IS NULL
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
    WHERE username=%s AND deleted_at IS NULL
    """
    with mysql_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (hashed_password, username))
            return cur.rowcount > 0


def update_user_password(username: str, hashed_password: str, *, cfg: Optional[MySQLConfig] = None) -> bool:
    return update_user_password_by_username(username=username, hashed_password=hashed_password, cfg=cfg)


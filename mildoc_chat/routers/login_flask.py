import logging
import os
from typing import Any, Dict, Optional

from flask import (
    Blueprint,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from mildoc_chat.routers.database import get_user_by_username

logger = logging.getLogger(__name__)

login_bp = Blueprint("mildoc_chat_login", __name__)

# 兼容旧的环境变量账号（可选）
CHAT_USERNAME = os.getenv("CHAT_USERNAME")
CHAT_PASSWORD = os.getenv("CHAT_PASSWORD")


@login_bp.get("/login")
def login():
    if "chat_username" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@login_bp.post("/login")
def login_post():
    data = request.form or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    user: Optional[Dict[str, Any]] = None
    try:
        user = get_user_by_username(username)
    except Exception:  # noqa: BLE001
        logger.exception("db get_user_by_username failed")
        return render_template(
            "login.html",
            error_message="数据库连接失败，请检查本地 MySQL 配置",
            last_username=username,
        )

    if user and isinstance(user.get("hashed_password"), str):
        from werkzeug.security import check_password_hash

        stored = user.get("hashed_password") or ""
        ok = False
        try:
            # 优先处理 werkzeug 生成的 pbkdf2/scrypt 等哈希
            if stored.startswith(("pbkdf2:", "scrypt:")):
                ok = check_password_hash(stored, password)
            # 兼容 passlib 生成的 bcrypt 哈希（形如 $2b$...）
            elif stored.startswith(("$2a$", "$2b$", "$2y$")):
                try:
                    from passlib.context import CryptContext

                    bcrypt_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
                    ok = bcrypt_ctx.verify(password, stored)
                except Exception:  # noqa: BLE001
                    logger.exception("bcrypt verify failed for user=%s", username)
                    ok = False
            else:
                # 其他情况按 werkzeug 默认格式尝试一次
                ok = check_password_hash(stored, password)
        except Exception:  # noqa: BLE001
            logger.exception("check_password failed for user=%s", username)
            ok = False

        if ok:
            session["chat_username"] = username
            return redirect(url_for("index"))
        return render_template("login.html", error_message="用户名或密码错误", last_username=username)

    if CHAT_USERNAME and CHAT_PASSWORD and username == CHAT_USERNAME and password == CHAT_PASSWORD:
        session["chat_username"] = username
        return redirect(url_for("index"))

    return render_template("login.html", error_message="用户名或密码错误", last_username=username)


@login_bp.get("/logout")
def logout():
    session.pop("chat_username", None)
    return redirect(url_for("login"))


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

from agribot_chat.routers.database import (
    get_user_by_username,
    get_user_by_email,
)

logger = logging.getLogger(__name__)

login_bp = Blueprint("agribot_chat_login", __name__)


@login_bp.get("/login")
def login():
    if "chat_username" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@login_bp.post("/login")
def login_post():
    data = request.form or {}
    identifier = (data.get("username") or "").strip()  # 可以是用户名或邮箱
    password = (data.get("password") or "").strip()

    if not identifier or not password:
        return render_template(
            "login.html",
            error_message="账号和密码均不能为空",
            last_username=identifier,
        )

    user: Optional[Dict[str, Any]] = None
    try:
        # 简单规则：包含 @ 时优先按邮箱查，否则按用户名查
        if "@" in identifier:
            user = get_user_by_email(identifier)
        if not user:
            user = get_user_by_username(identifier)
    except Exception:  # noqa: BLE001
        logger.exception("db get_user_by_username failed")
        return render_template(
            "login.html",
            error_message="数据库连接失败，请检查本地 MySQL 配置",
            last_username=identifier,
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
                    logger.exception("bcrypt verify failed for user=%s", user.get("username"))
                    ok = False
            else:
                # 其他情况按 werkzeug 默认格式尝试一次
                ok = check_password_hash(stored, password)
        except Exception:  # noqa: BLE001
            logger.exception("check_password failed for user=%s", user.get("username"))
            ok = False

        if ok:
            session["chat_username"] = user.get("username") or identifier
            return redirect(url_for("index"))
        return render_template("login.html", error_message="账号或密码错误", last_username=identifier)

    return render_template("login.html", error_message="账号或密码错误", last_username=identifier)


@login_bp.get("/logout")
def logout():
    session.pop("chat_username", None)
    return redirect(url_for("agribot_chat_login.login"))


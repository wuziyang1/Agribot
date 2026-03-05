import json
import logging
import os
import re
import secrets
import smtplib
import time
from email.message import EmailMessage
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, jsonify, redirect, render_template, request, session, url_for

from mildoc_chat.routers.database import create_user, get_user_by_email, get_user_by_username

logger = logging.getLogger(__name__)

register_bp = Blueprint("mildoc_chat_register", __name__)

_register_codes: Dict[str, Dict[str, Any]] = {}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_username(username: str) -> Optional[str]:
    u = (username or "").strip()
    if not u:
        return "用户名不能为空"
    if len(u) < 3 or len(u) > 32:
        return "用户名长度需在 3-32 位之间"
    for ch in u:
        if not (ch.isalnum() or ch in "._-"):
            return "用户名仅支持字母、数字、._-"
    return None


def validate_password(password: str) -> Optional[str]:
    p = (password or "").strip()
    if len(p) < 6:
        return "密码至少 6 位"
    return None


def validate_email(email: str) -> Optional[str]:
    e = (email or "").strip()
    if not e:
        return "邮箱不能为空"
    if len(e) > 200 or not _EMAIL_RE.match(e):
        return "邮箱格式不正确"
    return None


def send_email_code(email: str, code: str, *, scene: str = "register") -> Tuple[bool, str]:
    """发送验证码。未配置 SMTP 时，只写入日志并返回成功（便于内网/开发环境）。"""
    smtp_host = (os.getenv("SMTP_SERVER") or os.getenv("MILDOC_CHAT_SMTP_HOST") or "").strip()
    smtp_port = int(os.getenv("SMTP_PORT") or os.getenv("MILDOC_CHAT_SMTP_PORT") or "587")
    smtp_user = (os.getenv("SMTP_USERNAME") or os.getenv("MILDOC_CHAT_SMTP_USER") or "").strip()
    smtp_password = (os.getenv("SMTP_PASSWORD") or os.getenv("MILDOC_CHAT_SMTP_PASSWORD") or "").strip()
    smtp_from = (
        os.getenv("FROM_EMAIL")
        or os.getenv("MILDOC_CHAT_SMTP_FROM")
        or smtp_user
    ).strip()

    if not smtp_host or not smtp_from:
        logger.info("%s code (no smtp) email=%s code=%s", scene, email, code)
        return True, "未配置邮箱服务，验证码已写入服务日志"

    try:
        msg = EmailMessage()
        subject = "Mildoc Chat 注册验证码" if scene == "register" else "Mildoc Chat 找回密码验证码"
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = email
        msg.set_content(
            f"你的 Mildoc Chat { '注册' if scene == 'register' else '找回密码' }验证码是：{code}\n\n有效期 10 分钟。"
        )

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            try:
                server.starttls()
            except Exception:  # noqa: BLE001
                pass
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return True, "验证码已发送"
    except Exception as e:  # noqa: BLE001
        logger.exception("send email failed")
        return False, f"发送失败：{e}"


@register_bp.get("/register")
def register():
    if "chat_username" in session:
        return redirect(url_for("index"))
    return render_template("register.html")


@register_bp.post("/register")
def register_post():
    data = request.form or {}
    email = (data.get("email") or "").strip()
    email_code = (data.get("email_code") or "").strip()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    err = validate_email(email)
    if err:
        return render_template("register.html", error_message=err, last_email=email, last_username=username)
    err = validate_username(username)
    if err:
        return render_template("register.html", error_message=err, last_email=email, last_username=username)
    err = validate_password(password)
    if err:
        return render_template("register.html", error_message=err, last_email=email, last_username=username)

    # 校验邮箱验证码
    key = email.lower()
    rec = _register_codes.get(key) or {}
    now = time.time()
    if not rec or rec.get("expires_at", 0) < now:
        return render_template("register.html", error_message="验证码已过期，请重新获取", last_email=email, last_username=username)
    if str(rec.get("code") or "") != email_code:
        return render_template("register.html", error_message="验证码错误", last_email=email, last_username=username)

    try:
        if get_user_by_username(username):
            return render_template("register.html", error_message="用户名已存在", last_email=email, last_username=username)
        if get_user_by_email(email):
            return render_template("register.html", error_message="该邮箱已被绑定", last_email=email, last_username=username)
    except Exception as e:  # noqa: BLE001
        logger.exception("db precheck failed")
        return render_template("register.html", error_message=f"数据库访问失败：{e}", last_email=email, last_username=username)

    from werkzeug.security import generate_password_hash

    try:
        create_user(username=username, email=email, hashed_password=generate_password_hash(password))
    except ValueError as e:
        return render_template("register.html", error_message=str(e), last_email=email, last_username=username)
    except Exception as e:  # noqa: BLE001
        logger.exception("db create_user failed")
        return render_template("register.html", error_message=f"注册失败：{e}", last_email=email, last_username=username)

    # 注册成功后清理验证码并直接登录
    _register_codes.pop(key, None)
    session["chat_username"] = username
    return redirect(url_for("index"))


@register_bp.post("/api/register/send_code")
def api_register_send_code():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip()
    err = validate_email(email)
    if err:
        return jsonify({"success": False, "error_message": err})

    key = email.lower()
    now = time.time()
    rec = _register_codes.get(key) or {}
    last_sent = float(rec.get("sent_at") or 0)
    cooldown = 60
    if now - last_sent < cooldown:
        return jsonify({"success": True, "cooldown_seconds": int(cooldown - (now - last_sent))})

    code = f"{secrets.randbelow(1000000):06d}"
    ok, msg = send_email_code(email, code, scene="register")
    if not ok:
        return jsonify({"success": False, "error_message": msg})

    _register_codes[key] = {
        "code": code,
        "sent_at": now,
        "expires_at": now + 600,
    }
    return jsonify({"success": True, "cooldown_seconds": cooldown})


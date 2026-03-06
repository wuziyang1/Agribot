"""
独立的找回密码后端脚本（仅依赖 SMTP + 当前项目的 MySQL users 表）

提供两个核心函数：
1. send_reset_code(email): 发送验证码到绑定邮箱
2. reset_password(email, verification_code, new_password): 校验验证码并重置密码
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import os
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from werkzeug.security import generate_password_hash

from mildoc_chat.routers.database import (
    get_user_by_email,
    update_user_password_by_username,
)


# 验证码缓存（进程内内存，生产建议改为 Redis 等集中存储）
# key = email.lower()
_CODES: Dict[str, Dict[str, Any]] = {}

# 验证码有效期（分钟）
CODE_EXPIRE_MINUTES = 10


def _generate_code() -> str:
    """生成 6 位数字验证码。"""
    return f"{random.randint(100000, 999999)}"


def _load_smtp_config() -> Tuple[str, int, str, str, str]:
    """从环境变量加载 SMTP 配置。"""
    server = os.getenv("SMTP_SERVER", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME", "")
    password = os.getenv("SMTP_PASSWORD", "")
    from_email = os.getenv("FROM_EMAIL", username)
    return server, port, username, password, from_email


def _send_email_via_smtp(to_email: str, code: str) -> bool:
    """
    通过 SMTP 发送验证码邮件。

    返回 True 表示发送成功；在“开发模式”下，如果未配置完整 SMTP，也会返回 True，
    但仅在控制台打印验证码，方便调试。
    """
    import socket

    smtp_server, smtp_port, smtp_user, smtp_password, from_email = _load_smtp_config()

    # 开发模式：配置不完整时仅打印验证码
    if not smtp_server or not smtp_user or not smtp_password:
        print("=" * 60)
        print("[开发模式] SMTP 未完整配置，验证码仅打印到控制台")
        print(f"收件人: {to_email}")
        print(f"验证码: {code}")
        print(f"有效期: {CODE_EXPIRE_MINUTES} 分钟")
        print("=" * 60)
        print("如需真实发邮件，请在 .env 中配置：")
        print("  SMTP_SERVER / SMTP_PORT / SMTP_USERNAME / SMTP_PASSWORD / FROM_EMAIL")
        return True

    try:
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = "Mildoc Chat 找回密码验证码"

        body = (
            f"您正在申请重置 Mildoc Chat 登录密码，验证码为：{code}\n\n"
            f"验证码有效期为 {CODE_EXPIRE_MINUTES} 分钟，请及时使用。\n\n"
            "如果这不是您的操作，请忽略本邮件。"
        )
        msg.attach(MIMEText(body, "plain", "utf-8"))

        socket.setdefaulttimeout(30)

        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            server.starttls()

        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()

        print(f"[找回密码] 验证码邮件已发送至: {to_email}")
        return True
    except socket.gaierror as e:
        print(f"[找回密码] 发送邮件失败：无法连接 SMTP {smtp_server}:{smtp_port}，错误：{e}")
        print(f"[开发模式] 验证码为 {code} (收件人: {to_email})")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[找回密码] 发送邮件失败：{e}")
        print(f"[错误] 验证码已生成但未成功发送：{code} (收件人: {to_email})")
        return False


def send_reset_code(email: str) -> Dict[str, Any]:
    """
    发送找回密码验证码（仅根据邮箱）。

    业务规则：
    - 必须存在该邮箱对应的用户（users.email）
    - 每个邮箱一定时间内重复发送会被覆盖旧验证码（简单实现）
    """
    email = (email or "").strip()
    if not email:
        return {"success": False, "error_message": "邮箱不能为空"}

    user = get_user_by_email(email)
    if not user:
        return {"success": False, "error_message": "该邮箱未绑定任何用户"}

    username = (user.get("username") or "").strip()
    if not username:
        return {"success": False, "error_message": "该邮箱绑定的用户信息不完整"}

    code = _generate_code()
    expires_at = datetime.now() + timedelta(minutes=CODE_EXPIRE_MINUTES)

    key = email.strip().lower()
    _CODES[key] = {
        "code": code,
        "username": username,
        "expires_at": expires_at,
    }

    ok = _send_email_via_smtp(email, code)
    if not ok:
        _CODES.pop(key, None)
        return {"success": False, "error_message": "验证码邮件发送失败，请稍后重试"}

    return {
        "success": True,
        "message": "验证码已发送到邮箱，请查收",
        "expires_in_minutes": CODE_EXPIRE_MINUTES,
    }


def reset_password(email: str, verification_code: str, new_password: str) -> Dict[str, Any]:
    """
    使用邮箱 + 验证码重置密码。

    返回结构示例：
      {"success": True} 或 {"success": False, "error_message": "..."}
    """
    email = (email or "").strip()
    verification_code = (verification_code or "").strip()
    new_password = new_password or ""

    if not email or not verification_code or not new_password:
        return {"success": False, "error_message": "邮箱、验证码和新密码均不能为空"}

    user = get_user_by_email(email)
    if not user:
        return {"success": False, "error_message": "该邮箱未绑定任何用户"}

    username = (user.get("username") or "").strip()
    if not username:
        return {"success": False, "error_message": "该邮箱绑定的用户信息不完整"}

    key = email.strip().lower()
    rec = _CODES.get(key)
    now = datetime.now()

    if not rec or now > rec.get("expires_at", now):
        return {"success": False, "error_message": "验证码不存在或已过期，请重新获取"}

    if rec.get("username") != username:
        return {"success": False, "error_message": "验证码与账户信息不匹配"}

    if str(rec.get("code") or "") != verification_code:
        return {"success": False, "error_message": "验证码错误"}

    try:
        hashed = generate_password_hash(new_password)
        ok = update_user_password_by_username(username=username, hashed_password=hashed)
        if not ok:
            return {"success": False, "error_message": "密码更新失败：用户不存在或已被删除"}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error_message": f"密码更新失败：{e}"}
    finally:
        _CODES.pop(key, None)

    return {"success": True}


if __name__ == "__main__":
    # 简单命令行用法示例：
    # 1) 发送验证码：python forgot_password_smtp.py send iswuziyang@163.com
    # 2) 重置密码：python forgot_password_smtp.py reset iswuziyang@163.com 123456 新密码
    import sys

    if len(sys.argv) < 3:
        print("用法：")
        print("  发送验证码: python forgot_password_smtp.py send <email>")
        print("  重置密码:   python forgot_password_smtp.py reset <email> <code> <new_password>")
        raise SystemExit(1)

    action = sys.argv[1]
    if action == "send":
        email_arg = sys.argv[2]
        print(send_reset_code(email_arg))
    elif action == "reset":
        if len(sys.argv) < 5:
            print("重置密码用法: python forgot_password_smtp.py reset <email> <code> <new_password>")
            raise SystemExit(1)
        email_arg = sys.argv[2]
        code_arg = sys.argv[3]
        new_pwd_arg = sys.argv[4]
        print(reset_password(email_arg, code_arg, new_pwd_arg))
    else:
        print(f"未知操作: {action}")
        raise SystemExit(1)


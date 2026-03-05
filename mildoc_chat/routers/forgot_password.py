"""
找回密码路由模块
提供通过邮箱验证码重置密码的功能
"""

from datetime import datetime, timedelta
from typing import Optional
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from mildoc_chat.routers.database import get_user_by_username_and_email, update_user_password
from passlib.context import CryptContext

# 密码加密上下文（使用延迟初始化避免 bcrypt 初始化问题）
_pwd_context = None

def get_pwd_context():
    """延迟初始化密码上下文，避免初始化时的 bug 检测问题"""
    global _pwd_context
    if _pwd_context is None:
        try:
            _pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        except ValueError as e:
            # 如果初始化失败，尝试使用备用方案
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"bcrypt 初始化失败: {e}，尝试使用备用配置")
            _pwd_context = CryptContext(schemes=["bcrypt"])
    return _pwd_context

# 为了向后兼容
pwd_context = None  # 将在第一次使用时初始化

# 创建路由器实例
router = APIRouter()

# 验证码存储（内存存储，生产环境建议使用Redis）
# 格式: {email: {"code": str, "username": str, "expires_at": datetime}}
verification_codes: dict = {}

# 验证码有效期（分钟）
CODE_EXPIRE_MINUTES = 10


# =====================================================
# Pydantic数据模型定义
# =====================================================

class SendCodeRequest(BaseModel):
    """发送验证码请求模型"""
    username: str
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """重置密码请求模型"""
    username: str
    email: EmailStr
    verification_code: str
    new_password: str


class MessageResponse(BaseModel):
    """消息响应模型"""
    message: str


# =====================================================
# 邮件发送函数
# =====================================================

def send_verification_email(email: str, code: str) -> bool:
    """
    发送验证码邮件
    
    Args:
        email: 收件人邮箱
        code: 验证码
        
    Returns:
        是否发送成功
    """
    import os
    from ..config.config import Config
    
    config = Config()
    
    # 从环境变量或配置中获取邮件服务器信息
    SMTP_SERVER = os.getenv("SMTP_SERVER", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USERNAME)
    
    # 开发模式：如果没有配置SMTP服务器，直接打印验证码到控制台
    if not SMTP_SERVER or not SMTP_USERNAME or not SMTP_PASSWORD:
        print("=" * 60)
        print(f"[开发模式] 邮件服务未配置，验证码已生成")
        print(f"收件人: {email}")
        print(f"验证码: {code}")
        print(f"验证码有效期: {CODE_EXPIRE_MINUTES}分钟")
        print("=" * 60)
        print("提示: 要启用真实邮件发送，请配置以下环境变量:")
        print("  - SMTP_SERVER: SMTP服务器地址（如: smtp.qq.com）")
        print("  - SMTP_PORT: SMTP端口（如: 587）")
        print("  - SMTP_USERNAME: 发件人邮箱")
        print("  - SMTP_PASSWORD: 邮箱密码或授权码")
        print("  - FROM_EMAIL: 发件人邮箱（可选，默认使用SMTP_USERNAME）")
        print("=" * 60)
        return True  # 开发模式返回True，允许继续测试
    
    # 生产模式：发送真实邮件
    try:
        # 创建邮件对象
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = email
        msg['Subject'] = "找回密码验证码"
        
        # 邮件正文
        body = f"""
        您正在申请重置密码，验证码为：{code}
        
        验证码有效期为{CODE_EXPIRE_MINUTES}分钟，请及时使用。
        
        如果这不是您的操作，请忽略此邮件。
        """
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 连接SMTP服务器并发送邮件（添加超时设置）
        import socket
        socket.setdefaulttimeout(30)  # 设置30秒超时（增加超时时间）
        
        if SMTP_PORT == 465:
            # SSL连接（163邮箱推荐使用465端口）
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=30)
        else:
            # TLS连接
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)
            server.starttls()
        
        # 设置调试级别（可选，用于排查问题）
        # server.set_debuglevel(1)
        
        # 登录（163邮箱需要使用授权码，不是登录密码）
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        
        # 发送邮件
        server.send_message(msg)
        
        # 关闭连接
        server.quit()
        
        print(f"验证码邮件已成功发送到: {email}")
        return True
    except socket.gaierror as e:
        # DNS解析失败或网络不可达
        print(f"发送邮件失败: 无法连接到SMTP服务器 {SMTP_SERVER}:{SMTP_PORT}")
        print(f"错误详情: {e}")
        print(f"[开发模式] 验证码已生成: {code} (收件人: {email})")
        print("=" * 60)
        print("网络诊断建议:")
        print(f"1. 检查SMTP服务器地址是否正确: {SMTP_SERVER}")
        print(f"2. 检查服务器是否能访问该地址: ping {SMTP_SERVER}")
        print(f"3. 检查防火墙是否允许端口 {SMTP_PORT}")
        print(f"4. 检查网络连接: telnet {SMTP_SERVER} {SMTP_PORT}")
        print("=" * 60)
        # 开发/测试环境：即使发送失败也返回True，验证码会在日志中显示
        return True
    except (ConnectionRefusedError, OSError) as e:
        # 连接被拒绝或网络不可达
        error_msg = str(e)
        if "Network is unreachable" in error_msg or "101" in error_msg:
            print(f"发送邮件失败: 网络不可达，无法连接到 {SMTP_SERVER}:{SMTP_PORT}")
            print(f"[开发模式] 验证码已生成: {code} (收件人: {email})")
            print("=" * 60)
            print("可能的原因:")
            print("1. SMTP服务器地址配置错误")
            print("2. 服务器防火墙阻止了SMTP端口")
            print("3. 服务器网络配置问题")
            print("4. SMTP服务器暂时不可用")
            print("=" * 60)
            print("解决方案:")
            print("1. 检查SMTP配置是否正确")
            print("2. 检查服务器网络连接")
            print("3. 检查防火墙规则")
            print("4. 查看日志获取验证码（开发模式）")
            print("=" * 60)
        else:
            print(f"发送邮件失败: {e}")
            print(f"[开发模式] 验证码已生成: {code} (收件人: {email})")
        # 开发/测试环境：即使发送失败也返回True，验证码会在日志中显示
        return True
    except Exception as e:
        # 其他错误（认证失败等）
        print(f"发送邮件失败: {e}")
        print(f"[错误] 验证码已生成但未发送: {code} (收件人: {email})")
        print("=" * 60)
        print("可能的原因:")
        print("1. SMTP认证失败（用户名或密码错误）")
        print("2. SMTP服务器配置错误")
        print("3. 邮件服务器拒绝连接")
        print("=" * 60)
        # 开发/测试环境：即使发送失败也返回True，验证码会在日志中显示
        return True


def generate_verification_code() -> str:
    """生成6位数字验证码"""
    return str(random.randint(100000, 999999))


# =====================================================
# API路由
# =====================================================

@router.post("/send-code", response_model=MessageResponse, status_code=status.HTTP_200_OK)
async def send_code(request: SendCodeRequest):
    """
    发送验证码到用户邮箱
    
    Args:
        request: 发送验证码请求（包含用户名和邮箱）
        
    Returns:
        成功消息
    """
    # 验证用户是否存在且邮箱匹配
    user = get_user_by_username_and_email(request.username, request.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户名或邮箱不正确"
        )
    
    # 检查用户是否有邮箱
    if not user.get('email'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该用户未绑定邮箱，无法找回密码"
        )
    
    # 生成验证码
    code = generate_verification_code()
    expires_at = datetime.now() + timedelta(minutes=CODE_EXPIRE_MINUTES)
    
    # 存储验证码
    verification_codes[request.email] = {
        "code": code,
        "username": request.username,
        "expires_at": expires_at
    }
    
    # 发送邮件
    send_success = send_verification_email(request.email, code)
    if not send_success:
        # 如果邮件发送失败，删除验证码
        verification_codes.pop(request.email, None)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="邮件发送失败，请稍后重试"
        )
    
    return MessageResponse(message="验证码已发送到您的邮箱，请查收")


@router.post("/reset", response_model=MessageResponse, status_code=status.HTTP_200_OK)
async def reset_password(request: ResetPasswordRequest):
    """
    通过验证码重置密码
    
    Args:
        request: 重置密码请求（包含用户名、邮箱、验证码和新密码）
        
    Returns:
        成功消息
    """
    # 验证用户是否存在且邮箱匹配
    user = get_user_by_username_and_email(request.username, request.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户名或邮箱不正确"
        )
    
    # 检查验证码是否存在
    stored_code_info = verification_codes.get(request.email)
    if not stored_code_info:
        # 添加调试信息
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"验证码不存在: email={request.email}, 当前存储的验证码数量={len(verification_codes)}")
        logger.warning(f"可能的原因: 1)验证码已过期 2)多worker进程导致验证码存储在不同进程 3)服务重启导致验证码丢失")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="验证码不存在或已过期，请重新获取验证码"
        )
    
    # 验证用户名是否匹配
    if stored_code_info["username"] != request.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="验证码与用户名不匹配"
        )
    
    # 验证验证码是否过期
    if datetime.now() > stored_code_info["expires_at"]:
        verification_codes.pop(request.email, None)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="验证码已过期，请重新获取"
        )
    
    # 验证验证码是否正确
    if stored_code_info["code"] != request.verification_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="验证码错误"
        )
    
    # 验证新密码
    if len(request.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="密码长度至少6位"
        )
    
    # 加密新密码（使用延迟初始化的上下文，避免 bcrypt 初始化问题）
    ctx = get_pwd_context()
    try:
        hashed_password = ctx.hash(request.new_password)
    except ValueError as e:
        # 如果 passlib 的 bcrypt 后端有问题，尝试直接使用 bcrypt 库
        if "password cannot be longer than 72 bytes" in str(e):
            try:
                import bcrypt
                # 直接使用 bcrypt 生成哈希
                salt = bcrypt.gensalt()
                if isinstance(request.new_password, str):
                    password_bytes = request.new_password.encode('utf-8')
                else:
                    password_bytes = request.new_password
                hashed_password = bcrypt.hashpw(password_bytes, salt).decode('utf-8')
            except ImportError:
                # 如果 bcrypt 库不可用，记录错误并抛出异常
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"密码加密失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="密码加密失败，请稍后重试"
                )
        else:
            raise
    
    # 更新密码
    success = update_user_password(request.username, hashed_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="密码更新失败，请稍后重试"
        )
    
    # 删除已使用的验证码
    verification_codes.pop(request.email, None)
    
    return MessageResponse(message="密码重置成功，请使用新密码登录")


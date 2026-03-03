import logging
import os
from logging.handlers import TimedRotatingFileHandler


def setup_logging(
    log_path: str,
    level: int = logging.INFO,
    when: str = "D",
    interval: int = 1,
    backup_count: int = 14,
) -> None:
    """
    初始化统一日志：
    - 输出到文件（按天轮转，保留 backup_count 份）
    - 同时输出到控制台（便于 nohup 重定向/容器日志）
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = TimedRotatingFileHandler(
        filename=log_path,
        when=when,
        interval=interval,
        backupCount=backup_count,
        encoding="utf-8",
        utc=False,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(fmt)

    # 避免重复添加 handler（热重载或重复调用时常见）
    def _same_handler_exists(handler_cls, target_file: str | None = None) -> bool:
        for h in root.handlers:
            if isinstance(h, handler_cls):
                if target_file is None:
                    return True
                if getattr(h, "baseFilename", None) == os.path.abspath(target_file):
                    return True
        return False

    if not _same_handler_exists(TimedRotatingFileHandler, log_path):
        root.addHandler(file_handler)
    if not _same_handler_exists(logging.StreamHandler):
        root.addHandler(stream_handler)


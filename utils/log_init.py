import os
import sys
import platform
import logging
from conf.service_args import service_config
from loguru import logger


class InterceptHandler(logging.Handler):
    def emit(self, record):
        level = service_config["Log"]["level"]

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def push_uvicorn_log():
    """
    将uvicorn日志装置到loguru日志中
    """
    loggers = ("uvicorn", "uvicorn.access")
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in loggers:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=service_config["Log"]["level"])]


def push_gunicorn_log():
    """
    将gunicorn日志装置到loguru日志中
    """
    loggers = ("gunicorn", "gunicorn.access")
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in loggers:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=service_config["Log"]["level"])]


# 配置日志路径
log_root = service_config["Log"]["windows_save_root"] if platform.system() == "Windows" \
    else service_config["Log"]["linux_save_root"]
log_path = os.path.join(log_root, service_config["ServiceName"], service_config["ServiceName"])
# 配置日志格式
log_format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | Process: {process} | Thread:{thread} | {file}:{line} | {" \
             "message}"
log_config = {"rotation": "00:00:00", "format": log_format, "enqueue": True,
              "retention": service_config["Log"]["retention_time"], "compression": "zip"}
# 设置日志输出
logger.add("{}_{{time:YYYY-MM-DD}}.log".format(log_path), backtrace=True, diagnose=True,
           level=service_config["Log"]["level"], **log_config)


if __name__ == "__main__":
    logger.debug('this is a debug message')
    logger.info('this is info message')
    logger.success('this is success message!')
    logger.warning('this is warning message')
    logger.error('this is error message')
    logger.critical('this is critical message!')

import time
import importlib
import socket
import os
import re
from datetime import datetime
from utils.log_init import logger
from conf.service_args import service_config, open_models
from utils.file_utils import get_import_path, check_dirs_in_path
from pydantic import BaseModel


class BaseResponseItem(BaseModel):
    cost: float
    message: str
    status: int
    timestamp: str


def add_time_info(function_name):
    def func_in(*args, **kwargs):
        start_time = time.time()
        analysis_res = function_name(*args, **kwargs)
        used_time = time.time() - start_time
        res_json = {"data": analysis_res,
                    "cost": round(used_time, 2),
                    "status": 200,
                    "message": "ok",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        return res_json
    return func_in


def is_port_usable(port):
    """检查端口号是否可用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        sock.connect(("localhost", port))
        logger.error("端口：{} 被占用".format(port))
        return False
    except(socket.timeout, ConnectionRefusedError):
        logger.info("端口：{} 未被占用".format(port))
        return True
    finally:
        sock.close()


def camelize(string: str, uppercase_first_letter: bool = True) -> str:
    """
    Convert string to CamelCase.

    Examples::

        >>> camelize("device_type")
        'DeviceType'
        >>> camelize("device_type", False)
        'deviceType'

    :func:`camelize` can be thought of as a inverse of :func:`underscore`,
    although there are some cases where that does not hold::

        >>> camelize(underscore("IOError"))
        'IoError'

    :param uppercase_first_letter: if set to `True` :func:`camelize` converts
        strings to UpperCamelCase. If set to `False` :func:`camelize` produces
        lowerCamelCase. Defaults to `True`.
    """
    if uppercase_first_letter:
        return re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), string)
    else:
        return string[0].lower() + camelize(string)[1:]


def auto_include_router(app, router_root, module_switch=open_models, router_file_suffix="api.py",
                        router_attr="router"):
    """
    todo 更优雅的自动注册机制
    从router文件根目录遍历文件，找到相应的api文件中的router，根据开关自动注册
    """
    # 收集定义router的api文件
    for root, dirs, files in os.walk(router_root):
        for name in files:
            if name.endswith(router_file_suffix):
                api_path = os.path.join(root, name)
                router_switch = bool(check_dirs_in_path(module_switch, api_path))
                if not router_switch:
                    continue
                # 导入api文件
                lib_path = get_import_path(api_path)
                lib = importlib.import_module(lib_path)
                # 获取router
                if hasattr(lib, router_attr):
                    sub_router = getattr(lib, 'router')
                    # 注册路由
                    app.include_router(sub_router)
                    logger.info("路由文件：{} 注册成功".format(name))


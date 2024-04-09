import os
from conf.service_args import project_root
from utils.file_utils import read_yaml

config_file = os.path.join(project_root, "core/ocr/config.yaml")


def concat_model_path(config):
    key = "model_path"
    config["Det"][key] = os.path.join(project_root, config["Det"][key])
    config["Rec"][key] = os.path.join(project_root, config["Rec"][key])
    config["Cls"][key] = os.path.join(project_root, config["Cls"][key])
    return config


def init_args(config_path=config_file):
    config = read_yaml(config_path)
    config = concat_model_path(config)

    return config


# 初始化参数,使用需要用import导入，防止不必要的二次复制 例如 import **.config.**_args as **_args
ocr_args = init_args()

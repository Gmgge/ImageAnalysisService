import os
from conf.service_args import project_root
from utils.file_utils import read_yaml

config_file = os.path.join(project_root, "core/ocr/config.yaml")
det_model = os.path.join(project_root, "data/ocr/ch/det/zf_det.onnx")
cls_model = os.path.join(project_root, "data/ocr/ch/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx")
rec_model = os.path.join(project_root, "data/ocr/ch/rec/ch_ppocr_mobile_v2.0_rec_infer.onnx")


def concat_model_path(config):
    key = "model_path"
    config["Det"][key] = det_model
    config["Rec"][key] = rec_model
    config["Cls"][key] = cls_model
    return config


def init_args(config_path=config_file):
    config = read_yaml(config_path)
    config = concat_model_path(config)

    return config


# 初始化参数,使用需要用import导入，防止不必要的二次复制 例如 import **.config.**_args as **_args
ocr_args = init_args()
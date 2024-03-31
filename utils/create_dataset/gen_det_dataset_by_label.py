import copy
import json
import os
import cv2
import numpy as np
from tqdm import tqdm

label_template = {
    "version": "5.2.0.post4",
    "flags": {},
    "shapes": [],
    "imagePath": "",
    "imageData": None,
    "imageHeight": 0,
    "imageWidth": 0
}

obj_shape_template = {
      "label": "round_seal",
      "points": [],
      "group_id": None,
      "description": "",
      "shape_type": "rectangle",
      "flags": {}
    }


def read_ppocr_label(label_path):
    """
    读取PPOCRLabel格式的ocr识别标签
    """
    data_root = os.path.dirname(label_path)
    with open(label_path, "r", encoding="utf-8") as f_r:
        labels_info = f_r.readlines()
        for one_info in tqdm(labels_info):
            image_path, label_data = one_info.split("\t")
            image_name = os.path.split(image_path)[-1]
            label_data = json.loads(label_data)
            image_path = os.path.join(data_root, image_name)
            yield image_path, label_data


def det_by_ppocr_label(label_path):
    """
    读取PPOCRLabel格式的印章识别标签，并根据标签生成目标检测训练集
    """
    for label_info in read_ppocr_label(label_path):
        image_path, label_data = label_info
        image_data = cv2.imread(image_path)
        if image_data is None:
            print("图像：{}，读取失败".format(image_path))
            continue
        image_h, image_w = image_data.shape[:2]
        # 生成检测标签模板 labelme格式 用于可视化检查并兼容多种进一步转换
        det_label = copy.deepcopy(label_template)
        det_label["imagePath"] = os.path.basename(image_path)
        det_label["imageHeight"] = image_h
        det_label["imageWidth"] = image_w
        # 获取印章box
        for one_obj in label_data:
            # 获取印章box
            obj_points = one_obj["points"]
            obj_label = copy.deepcopy(obj_shape_template)
            obj_label["points"] = [obj_points[0], obj_points[2]]
            det_label["shapes"].append(obj_label)
        # 存储图像 处理可能的通道问题
        cv2.imwrite(image_path, image_data)
        # 存储标签
        label_root = os.path.dirname(image_path)
        base_name, extension = os.path.splitext(image_path)
        seal_label = os.path.join(label_root, str(base_name) + ".json")
        with open(seal_label, "w", encoding="utf-8") as json_file:
            json.dump(det_label, json_file, indent=4)


if __name__ == "__main__":
    label_path_ = r"F:\project\seal\data\filter\Label.txt"
    save_dir_ = r"F:\dateset\image\seal\detect\image"
    det_by_ppocr_label(label_path_)


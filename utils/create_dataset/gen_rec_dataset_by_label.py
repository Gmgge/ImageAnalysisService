import json
import os
import cv2
import numpy as np
from tqdm import tqdm


def cut_by_ppocr_label(image_dir, label_path, save_dir):
    """
    读取PPOCRLabel格式的印章识别标签，并根据标签裁剪图像
    """
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_index = 0
    with open(label_path, "r", encoding="utf-8") as f_r:
        labels_info = f_r.readlines()
        for one_info in tqdm(labels_info):
            image_path, label_data = one_info.split("\t")
            image_name = os.path.split(image_path)[-1]
            label_data = json.loads(label_data)
            for one_obj in label_data:
                # 获取印章box
                seal_name = one_obj["transcription"]
                if len(seal_name) <= 0:
                    continue
                seal_points = np.array(one_obj["points"])
                x, y, w, h = cv2.boundingRect(seal_points)
                # 获取印章图像
                image_path = os.path.join(image_dir, image_name)
                image_data = cv2.imread(image_path)
                crop_image = image_data[y:y + h, x:x + w]
                # 存储图像
                save_path = os.path.join(save_dir, str(save_index) + ".jpg")
                cv2.imwrite(save_path, crop_image)
                # 存储标签
                seal_txt = os.path.join(save_dir, str(save_index) + ".txt")
                with open(seal_txt, "w", encoding="utf-8") as f_w:
                    f_w.write(seal_name)
                save_index += 1


if __name__ == "__main__":
    image_dir_ = r"F:\project\seal\data\filter"
    save_path_ = r"F:\project\seal\data\filter_cut"
    label_path_ = r"F:\project\seal\data\filter\Label.txt"
    cut_by_ppocr_label(image_dir_, label_path_, save_path_)

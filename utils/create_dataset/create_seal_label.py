import os
import json
from utils.image_utils import read_image_file, check_image_file
from core.seal_rec.analysis_system import SealRec


def get_images_seal_label(image_dir, label_path):
    # 构建检测器
    opt = SealRec()
    # 图像获取
    images_path = [os.path.join(image_dir, one_path) for one_path in os.listdir(image_dir)]
    # 标签存储
    with open(label_path, "w", encoding="utf-8") as fw:
        for path in images_path:
            if not check_image_file(path):
                continue
            image_data = read_image_file(path)
            # 预测
            try:
                detect_res = opt.analysis(image_data)
            except Exception as e:
                print(e)
                print(path)
                continue
            # 收集具有印章的数据
            new_label_info = []
            # 依次处理识别的印章
            for one_seal in detect_res:
                temp_obj_info = {"transcription": one_seal["info"],
                                 "points": [[one_seal["box"][0], one_seal["box"][1]],
                                            [one_seal["box"][2], one_seal["box"][1]],
                                            [one_seal["box"][2], one_seal["box"][3]],
                                            [one_seal["box"][0], one_seal["box"][3]]], "difficult": "false"}
                new_label_info.append(temp_obj_info)
            image_name = "filter/{}".format(os.path.basename(path))
            new_label = "{}\t{}\n".format(image_name, new_label_info)
            fw.write(new_label)
            print(detect_res)


if __name__ == "__main__":
    images_root = r"F:\project\seal\data\filter"
    label_save_path = r"F:\project\seal\data\filter\label.cach"
    get_images_seal_label(images_root, label_save_path)

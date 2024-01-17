import os
import json
from utils.image_utils import read_image_file, check_image_file
from core.seal_rec.analysis_system import SealRec


def get_images_seal_label(image_dir):
    # 构建检测器
    opt = SealRec()
    # 图像获取
    images_path = [os.path.join(image_dir, one_path) for one_path in os.listdir(image_dir)]
    # 标签存储
    for path in images_path:
        if not check_image_file(path):
            break
        image_data = read_image_file(path)
        # 预测
        detect_res = opt.analysis(image_data)
        # 收集具有印章的数据
        print(detect_res)


if __name__ == "__main__":
    images_root = r"F:\project\seal\data\filter"
    get_images_seal_label(images_root)

import os
import shutil
from conf.service_args import project_root
from utils.image_utils import read_image_file
from core.seal_rec.detect.picodet import PicoDet


def filter_images(image_dir, save_dir):
    # 构建检测器
    net = PicoDet()
    # 图像读取
    images_path = [os.path.join(image_dir, one_path) for one_path in os.listdir(image_dir)]
    for path in images_path:
        image_data = read_image_file(path)
        # 预测
        detect_res = net.detect(image_data)
        # 收集具有印章的数据
        if len(detect_res) > 0:
            shutil.move(path, save_dir)


if __name__ == "__main__":
    dirs_root = r"F:\project\seal\data\爬取的原始数据"
    images_root = [os.path.join(dirs_root, one_path) for one_path in os.listdir(dirs_root)]
    for root_path in images_root:
        save_dir = root_path+"filter"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        filter_images(root_path, save_dir)

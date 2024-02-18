

def cut_by_ppocr_label(image_dir, label_path):
    """
    读取PPOCRLabel格式的印章识别标签，并根据标签裁剪图像
    """
    with open(label_path, "r", encoding="utf-8") as f_r:
        label_data = f_r.readlines()


if __name__ == "__main__":
    image_dir_ = r"F:\project\seal\data\filter"
    label_path_ = r"F:\project\seal\data\filter\Label.txt"
    cut_by_ppocr_label(image_dir_, label_path_)
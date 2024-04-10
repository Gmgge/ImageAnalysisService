import base64

import numpy as np
import cv2
import os
import imghdr
from PIL import Image
from io import BytesIO
from utils.log_init import logger
from conf.global_constant import IMG_PATH, IMG_STREAM
from starlette.datastructures import UploadFile
from PIL import UnidentifiedImageError
from utils.web_exception import FileNotExistException, UnsupportedParametersException, UnsupportedFileTypeException


def check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    return any([path.lower().endswith(e) for e in img_end])


def read_image_file(image, mode=IMG_PATH):
    """
    读取图像信息,并根据推理需要转换为三通道图像，支持本地文件与form-data数据
    当image为form_data时，mode参数无效
    """
    # 本地文件检查
    if isinstance(image, str) and mode == IMG_PATH:
        if not os.path.isfile(image):
            raise FileNotExistException(image)
    # base64 读取
    elif isinstance(image, str) and mode == IMG_STREAM:
        binary_img_data = base64.b64decode(image)
        image = BytesIO(binary_img_data)
    # form_data 文件读取
    elif isinstance(image, UploadFile):
        image = BytesIO(image.file.read())
    # 其他形式数据
    else:
        err_info = "不支持该形式数据"
        raise UnsupportedParametersException(err_info)
    #
    try:
        image_pillow = Image.open(image).convert("RGB")
        image_data = np.array(image_pillow)
    except UnidentifiedImageError:
        raise UnsupportedFileTypeException("image")

    # 图像数据有效性检查
    if image_data is None:
        err_info = "不支持的图像数据"
        raise UnsupportedParametersException(err_info)
    return image_data


def resize_image(src_img, new_shape, keep_ratio=False):
    top, left, new_h, new_w = 0, 0, new_shape[0], new_shape[1]
    origin_shape = src_img.shape[:2]
    im_scale_y = new_h / float(origin_shape[0])
    im_scale_x = new_w / float(origin_shape[1])
    img_shape = np.array([[float(new_shape[0]), float(new_shape[1])]]).astype('float32')
    scale_factor = np.array([[im_scale_y, im_scale_x]]).astype('float32')
    if keep_ratio and src_img.shape[0] != src_img.shape[1]:
        hw_scale = src_img.shape[0] / src_img.shape[1]
        if hw_scale > 1:
            new_h, new_w = new_shape[0], int(new_shape[1] / hw_scale)
            img = cv2.resize(src_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            left = int((new_shape[1] - new_w) * 0.5)
            img = cv2.copyMakeBorder(img,
                                     0,
                                     0,
                                     left,
                                     new_shape[1] - new_w - left,
                                     cv2.BORDER_CONSTANT,
                                     value=(0, 0, 0))  # add border
        else:
            new_h, new_w = int(new_shape[0] * hw_scale), new_shape[1]
            img = cv2.resize(src_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            top = int((new_shape[0] - new_h) * 0.5)
            img = cv2.copyMakeBorder(img,
                                     top,
                                     new_shape[0] - new_h - top,
                                     0,
                                     0,
                                     cv2.BORDER_CONSTANT,
                                     value=(0, 0, 0))
    else:
        img = cv2.resize(src_img, tuple(new_shape), interpolation=cv2.INTER_LINEAR)

    return img, img_shape, scale_factor


def normalize(img, mean, std):
    """
    图像归一化
    """
    img = img.astype(np.float32)
    img = (img / 255.0 - mean) / std
    return img


def get_color_map_list(num_classes):
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map

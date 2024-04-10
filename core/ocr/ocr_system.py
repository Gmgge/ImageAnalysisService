import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
from libs.ocr_onnxruntime.ch_ppocr_v2_cls import TextClassifier
from libs.ocr_onnxruntime.ch_ppocr_v3_det import TextDetector
from libs.ocr_onnxruntime.ch_ppocr_v3_rec import TextRecognizer
from libs.ocr_onnxruntime.utils import LoadImage
from core.ocr.tool import compute_line_iou, text_line_angle, get_rotate_crop_image, sort_boxes, add_boxes_info, \
    rotate_image_and_boxes, rotate_points, rotate_boxes
from libs.base.image_analysis import BaseImageAnalysis
from core.ocr.config import ocr_args
from utils.log_init import logger


class OCR(BaseImageAnalysis):
    def __init__(self, module_args=ocr_args):
        super().__init__(module_args)
        self.print_verbose = module_args["Global"]["print_verbose"]
        self.text_score = module_args["Global"]["text_score"]
        self.min_height = module_args["Global"]["min_height"]
        self.width_height_ratio = module_args["Global"]["width_height_ratio"]

        self.text_det = TextDetector(module_args["Det"])
        self.text_cls = TextClassifier(module_args["Cls"])
        self.text_rec = TextRecognizer(module_args["Rec"])

        self.load_img = LoadImage()

    def analysis(self,
                 image: Union[str, np.ndarray, bytes, Path],
                 use_det: bool = True,
                 use_cls: bool = True,
                 use_rec: bool = True, ):
        img = self.load_img(image)
        # 初始化默认结果
        det_boxes, cls_res, rec_res = None, None, None
        ocr_res = ""
        det_elapse, cls_elapse, rec_elapse = 0.0, 0.0, 0.0
        img_list = []
        # 文字区域检测
        if use_det:
            det_boxes, det_elapse = self.auto_text_det(img)
            if det_boxes is None:
                return ocr_res
            # # 绘制文本行以进行debug
            # det_boxes = np.asarray(det_boxes).astype(np.int32)
            # det_boxes = det_boxes[:, :4, :]
            # img = cv2.drawContours(img, det_boxes, -1, (0, 255, 0), 2)7
            # cv2.imshow("image", img)
            # cv2.waitKey(0)
            # 增加文本行形心点用于排序、行合并
            det_boxes = add_boxes_info(det_boxes)
            # 根据文本行倾斜，纠正形心位置
            img, det_boxes = self.tilt_correction(img, det_boxes)
            # 文本行排序
            det_boxes = sort_boxes(det_boxes)
            # print(img.shape)
            # cv2.imshow("image", img)
            # cv2.waitKey(0)
            img_list = self.get_crop_img_list(img, det_boxes)
        # 文本行翻转分类
        flip_sign = False
        if use_cls:
            img_list, cls_res, flip_sign, cls_elapse = self.text_cls(img_list)
            # 列表翻转
            if flip_sign:
                img_list = np.flip(np.asarray(img_list, dtype=object), axis=0)
                det_boxes = np.flip(np.asarray(det_boxes), axis=0)
            # # 绘制文本行以进行debug
            # det_boxes = np.asarray(det_boxes).astype(np.int32)
            # det_boxes = det_boxes[:, :4, :]
            # img = cv2.drawContours(img, det_boxes, -1, (0, 255, 0), 2)
            # cv2.imshow("image", img)
            # cv2.waitKey(0)
            # for i in range(len(img_list)):
            #     print(det_boxes[i])
            #     cv2.imshow("test", img_list[i])
            #     cv2.waitKey(0)
        # 文本行识别
        if use_rec:
            rec_res, rec_elapse = self.text_rec(img_list)
        ocr_res = self.get_final_res(det_boxes, rec_res, flip_sign)
        logger.info(f"识别前部分结果为:{ocr_res[:5]}")
        return ocr_res

    def auto_text_det(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        h, w = img.shape[:2]
        # 文本行图像跳过检测
        if self.width_height_ratio == -1:
            use_limit_ratio = False
        else:
            use_limit_ratio = w / h > self.width_height_ratio
        if h <= self.min_height or use_limit_ratio:
            det_boxes = self.get_boxes_img_without_det(h, w)
            return det_boxes, 0.0
        # 图像文本行检测
        det_boxes, det_elapse = self.text_det(img)
        if det_boxes is None or len(det_boxes) < 1:
            return None, 0.0
        return np.asarray(det_boxes), float(det_elapse)

    @staticmethod
    def tilt_correction(img, det_boxes, angle_threshold=2):
        # 计算倾斜角度
        angle = text_line_angle(det_boxes)
        # 旋转图像与box
        if abs(angle) > angle_threshold:
            center_point = det_boxes[:, 4:, :]
            det_boxes[:, 4:, :] = rotate_points(img.shape, center_point, angle)
            # det_boxes = rotate_boxes(img.shape, det_boxes, angle)
            # img, det_boxes = rotate_image_and_boxes(img, det_boxes, angle)
        return img, det_boxes

    @staticmethod
    def get_boxes_img_without_det(h, w):
        x0, y0, x1, y1 = 0, 0, w, h
        det_boxes = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        det_boxes = det_boxes[np.newaxis, ...]
        return det_boxes

    @staticmethod
    def get_crop_img_list(img, det_boxes):
        img_crop_list = []
        for box in det_boxes:
            img_crop = get_rotate_crop_image(img, box)
            img_crop_list.append(img_crop)
        return img_crop_list

    def get_final_res(self, det_boxes, rec_res, flip_sign):
        ocr_res = ""
        if det_boxes is None or rec_res is None:
            return ocr_res
        det_boxes, rec_res = self.filter_result(det_boxes, rec_res)
        if len(rec_res) <= 0:
            return ocr_res
        ocr_res = self.row_format_recovery(det_boxes, rec_res, flip_sign)
        return ocr_res

    @staticmethod
    def row_format_recovery(det_boxes, rec_res, flip_sign):
        """
        根据文本行位置坐标，恢复文本行结果
        :param flip_sign: 翻转标志
        :param det_boxes:
        :param rec_res:
        :return:
        """
        text_res = ""
        if len(rec_res) < 1:
            return text_res
        pre_box = det_boxes[0]
        text_res += rec_res[0][0]
        # 根据box形心进行判断
        for index, box in enumerate(det_boxes[1:]):
            # y轴位置在误差范围内，x轴位置从左到右(翻转反之)，则归做一行
            pre_box_center_x = -pre_box[4][0] if flip_sign else pre_box[4][0]
            box_center_x = -box[4][0] if flip_sign else box[4][0]
            if abs(pre_box[4][1] - box[4][1]) < 10 and pre_box_center_x < box_center_x:
                text_res = text_res + " " + rec_res[index + 1][0]
            else:
                text_res = text_res + "\n" + rec_res[index + 1][0]
            pre_box = box
        return text_res

    def filter_result(self, det_boxes, rec_res):
        """
        根据阈值过滤文本行
        :param det_boxes:
        :param rec_res:
        :return:
        """
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(det_boxes, rec_res):
            text, score = rec_result
            if float(score) >= self.text_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        return filter_boxes, filter_rec_res


if __name__ == "__main__":
    import os
    from conf.service_args import project_root

    engine = OCR()
    dir_path = os.path.join(project_root, "tests/core_test/ocr_test/demo")
    for image_name in os.listdir(dir_path):
        image_path = r"D:\GitHubWorkSpace\RapidOcrOnnx\images\1.jpg"
        # image_path = os.path.join(dir_path, image_name)
        result = engine.analysis(image_path)
        print(result)
        exit(0)

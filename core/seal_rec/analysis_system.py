from core.seal_rec.detect.picodet import PicoDet
from core.seal_rec.rec.trocr import OnnxEncoderDecoder
from libs.base.image_analysis import BaseImageAnalysis
from core.seal_rec.config import seal_rec_args


class SealRec(BaseImageAnalysis):
    def __init__(self, module_args=seal_rec_args):
        super().__init__(module_args)
        self.seal_det_module, self.seal_rec_module = self.init_analysis_module()

    @staticmethod
    def init_analysis_module():
        seal_det_module = PicoDet()
        seal_rec_module = OnnxEncoderDecoder()
        return seal_det_module, seal_rec_module

    def analysis(self, image):
        seal_info = []
        # 印章检测
        boxes = self.seal_det_module.detect(image)
        for i in range(boxes.shape[0]):
            x_min, y_min, x_max, y_max = int(boxes[i, 2]), int(boxes[i, 3]), int(boxes[i, 4]), int(boxes[i, 5])
            one_seal_image = image[y_min:y_max:, x_min:x_max, :]
            # 印章识别
            one_info = self.seal_rec_module.rec(one_seal_image)
            # 印章识别结果过滤
            if len(one_info) > 0:
                one_seal_res = {"box": [x_min, y_min, x_max, y_max], "info": one_info}
                seal_info.append(one_seal_res)
        return seal_info


if __name__ == "__main__":
    import cv2
    import os
    from utils.image_utils import read_image_file

    test_opt = SealRec()
    # 图像读取
    image_root = r"./seal_image/"
    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        test_data = read_image_file(image_path)
        if test_data is None:
            print("{} image read err".format(image_name))
            continue
        # 预测
        res_info = test_opt.analysis(test_data)
        # 可视化结果
        if len(res_info) > 0:
            print("{} image rec :{}".format(image_name, res_info))

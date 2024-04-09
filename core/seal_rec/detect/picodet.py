import cv2
import numpy as np
import onnxruntime as ort
from utils.image_utils import get_color_map_list, normalize
from core.seal_rec.detect.config import seal_detect_args
from utils.compute.bbox_nms import multiclass_nms
from utils.image_utils import resize_image


class PicoDet(object):
    def __init__(self, module_args=seal_detect_args):
        # 读取模型信息
        self.classes = list(map(lambda x: x.strip(), open(module_args.label_path, 'r').readlines()))
        self.score_thresh = module_args.score_thresh
        self.nms_thresh = module_args.nms_thresh
        self.mean = np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape(1, 1, 3)
        # 设置推理配置
        options = ort.SessionOptions()
        options.log_severity_level = 3
        # 构建推理会话
        self.net = ort.InferenceSession(module_args.model_path, options)
        inputs_name = [a.name for a in self.net.get_inputs()]
        inputs_shape = {k: v.shape for k, v in zip(inputs_name, self.net.get_inputs())}
        self.input_shape = inputs_shape['image'][2:]
        # 绘制参数
        self.color_list = get_color_map_list(len(self.classes))

    def detect(self, src_img):
        # 数据预处理
        img, im_shape, scale_factor = resize_image(src_img, self.input_shape)
        img = normalize(img, self.mean, self.std)
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        inputs_dict = {'image': blob}
        inputs_name = [a.name for a in self.net.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}
        # 推理
        pred_boxes, pred_scores = self.net.run(None, net_inputs)[:2]
        # nms 过滤
        outs = multiclass_nms(pred_boxes, pred_scores, self.score_thresh, self.nms_thresh)
        boxes = np.array(outs[0])
        if len(boxes) > 0:
            # 过滤可能的背景类
            expect_boxes = boxes[:, 0] > -1
            boxes = boxes[expect_boxes, :]
            # 恢复坐标尺度
            boxes[:, 2], boxes[:, 4] = boxes[:, 2]/scale_factor[0][1], boxes[:, 4]/scale_factor[0][1]
            boxes[:, 3], boxes[:, 5] = boxes[:, 3]/scale_factor[0][0], boxes[:, 5]/scale_factor[0][0]
        return boxes

    def detect_and_draw(self, src_img):
        boxes = self.detect(src_img)
        for i in range(boxes.shape[0]):
            class_id, conf = int(boxes[i, 0]), boxes[i, 1]
            x_min, y_min, x_max, y_max = int(boxes[i, 2]), int(boxes[i, 3]), int(boxes[i, 4]), int(boxes[i, 5])
            color = tuple(self.color_list[class_id])
            cv2.rectangle(src_img, (x_min, y_min), (x_max, y_max), color, thickness=2)
            print(self.classes[class_id] + ': ' + str(round(conf, 3)))
            cv2.putText(src_img,
                        self.classes[class_id] + ':' + str(round(conf, 3)), (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0),
                        thickness=2)
        return src_img


if __name__ == '__main__':
    import os
    from tqdm import tqdm
    from utils.image_utils import read_image_file
    # 构建检测器
    net = PicoDet()
    # 图像读取
    test_dir = r"D:\work_data\official_doc_img\filter_group\99_img"
    for one_name in tqdm(os.listdir(test_dir)):
        test_image_path = os.path.join(test_dir, one_name)
        test_data = read_image_file(test_image_path)
        # 预测
        res_image = net.detect(test_data)
        # # 可视化结果
        # res_image = cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR)
        # cv2.imshow("detect_res", res_image)
        # cv2.waitKey(0)

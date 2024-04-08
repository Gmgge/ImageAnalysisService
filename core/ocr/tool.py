import numpy as np
import cv2 as cv
from functools import cmp_to_key
from math import fabs, sin, radians, cos


def box_position_compare(box_a, box_b):
    """
    针对近似矩形文本行进行从上到下，从左到右的排序
    :param box_a: [[x1, y1], [x1, y1], [x1, y1], [x1, y1], [center_x, center_y]]
    :param box_b: 与box_a相同组成结构
    :return: compare_sign  位置上 box_a>box_b,compare_sign为True,反之为False
    """
    position_bias = 10  # 预测出的文本行位置偏差
    if box_b[4][1] - box_a[4][1] >= position_bias or \
            (abs(box_b[4][1] - box_a[4][1]) < position_bias and box_b[4][0] > box_a[4][0]):
        compare_sign = -1
    elif box_a[4][0] == box_b[4][0] and box_a[4][1] == box_b[4][1]:
        compare_sign = 0
    else:
        compare_sign = 1
    return compare_sign


def add_boxes_info(det_boxes):
    """
    给预测出的文本行添加位置信息，例如box的形心等
    :param det_boxes: np.array 预测出的文本行
    :return:boxes_info,其结构为:[[box_info]], box_info:[[x1, y1], [x1, y1], [x1, y1], [x1, y1], [center_x, center_y]]
    """
    temp_boxes = det_boxes.reshape((-1, 8))
    centroids = np.apply_along_axis(compute_centroid, 1, temp_boxes)
    centroids_expanded = centroids[:, np.newaxis, :]
    boxes_info = np.concatenate((det_boxes, centroids_expanded), axis=1)
    return boxes_info


def sort_boxes(det_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        det_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    sorted_boxes = sorted(det_boxes, key=cmp_to_key(box_position_compare))
    return sorted_boxes


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    points = points[:4]
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0],
                          [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    rotation_matrix = cv.getPerspectiveTransform(points, pts_std)
    dst_img = cv.warpPerspective(img,
                                 rotation_matrix,
                                 (img_crop_width, img_crop_height),
                                 borderMode=cv.BORDER_REPLICATE,
                                 flags=cv.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def text_line_angle(det_boxes):
    """
    根据最大文本区域计算文本倾斜角度
    :param det_boxes:
    :return: 文本行倾斜角度
    """
    # 取boxes边框坐标区域
    det_boxes = det_boxes[:, :4]
    # 按照box周长排序
    temp_boxes = det_boxes.reshape((-1, 8))
    arcs_length = np.apply_along_axis(compute_arc_length, 1, temp_boxes)
    sorted_indices = np.argsort(arcs_length)
    # 取最长的文本行box
    represent_box = det_boxes[sorted_indices[-1]]
    # 计算文本行的角度
    angle = compute_box_angle(represent_box)
    return angle


def rotate_boxes(image_shape, boxes, angle, mode=1):
    """
    跟随图片，旋转矩形框列表
    :param image_shape:跟随图像的shape
    :param boxes: 矩形框列表 np.array
    :param angle: 旋转角度
    :param mode: 旋转模式，1:保留全部内容 0:尺度不变
    :return: 旋转后的点坐标
    """
    boxes_shape = boxes.shape
    points = boxes.reshape((-1, 2))
    points_rotate = rotate_points(image_shape, points, angle, mode)
    boxes_rotate = points_rotate.reshape(boxes_shape)
    return boxes_rotate


def rotate_points(image_shape, points, angle, mode=1):
    """
    跟随图片，旋转坐标点列表
    :param image_shape:跟随图像的shape
    :param points: 坐标的列表 np.array
    :param angle: 旋转角度
    :param mode: 旋转模式，1:保留全部内容 0:尺度不变
    :return: 旋转后的点坐标
    """
    # 计算旋转中心
    h, w = image_shape[:2]
    center = (w / 2, h / 2)
    points_shape = points.shape
    points = points.reshape((points_shape[0], 2))
    if mode == 1:
        new_h = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
        new_w = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
        points[:, 0] += (new_w - w) / 2
        points[:, 1] += (new_h - h) / 2
        center = (new_w / 2, new_h / 2)
    # 旋转坐标点
    angle = angle * np.pi / 180
    x = points[:, 0] - center[0]
    y = points[:, 1] - center[1]
    points[:, 0] = x * np.cos(angle) + y * np.sin(angle) + center[0]
    points[:, 1] = -x * np.sin(angle) + y * np.cos(angle) + center[1]
    points = points.reshape(points_shape)
    return points


def rotate_image_and_boxes(img_data, boxes, angle, mode=1):
    """
    旋转图像、图像中的boxes
    :param img_data: 图像矩阵 np.array
    :param boxes: 图像中的矩形敏感区域 np.array
    :param angle: 旋转角度 正值表示逆时针旋转（根据坐标系的旋转规则），负值表示顺时针旋转
    :param mode: 旋转模式，1:保留全部内容 0:尺度不变
    :return:
    """
    # 计算透视变换参数
    new_h, new_w = h, w = img_data.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    if mode == 1:
        new_h = int(w * abs(rotation_matrix[0][1]) + h * abs(rotation_matrix[0][0]))
        new_w = int(h * abs(rotation_matrix[0][1]) + w * abs(rotation_matrix[0][0]))
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
    # 变换图像
    img_rotate = cv.warpAffine(img_data, rotation_matrix, (new_w, new_h), flags=cv.INTER_AREA)
    # 变换boxes
    boxes_rotate = rotate_boxes(img_data.shape, boxes, angle, mode)
    return img_rotate, boxes_rotate


def compute_line_iou(line_0, line_1):
    """
    计算两个线段的交并比
    :param line_0: (start, end)
    :param line_1: (start, end)
    :return: iou float
    """
    lines_index = np.array([line_0[0], line_0[1], line_1[0], line_1[1]])
    # 计算交集
    union = max(np.ptp(lines_index), np.finfo(float).eps)  # 防止被除数为0
    # 计算并集
    intersection = max(0, abs(line_0[0] - line_0[1]) + abs(line_1[0] - line_1[1]) - union)
    iou = intersection / union
    return iou


def compute_arc_length(contours):
    """
    装饰cv2计算周长函数，以在np.apply_along_axis批量计算 contours 列表
    :param contours
    :return: 周长
    """
    contours = contours.reshape((-1, 2))
    return cv.arcLength(contours, True)


def compute_centroid(contours):
    """
    装饰cv2计算质心函数，以在np.apply_along_axis批量计算 contours 列表
    :param contours
    :return: 周长
    """
    contours = contours.reshape((-1, 2))
    # 计算质心
    M = cv.moments(contours)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])
    return np.array([centroid_x, centroid_y], dtype=np.float32)


def calculate_line_angle(point_1, point_2):
    # 计算两点之间的水平距离和垂直距离
    delta_x = point_2[0] - point_1[0]
    delta_y = point_2[1] - point_1[1]

    # 使用numpy的arctan2函数来计算角度，arctan2自动处理不同象限的情况
    angle_rad = np.arctan2(delta_y, delta_x)

    # 将弧度转换为度
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def compute_box_angle(text_box):
    """
    计算文本行倾斜角度，先筛选最长的对边(近似，因为box可能为接近矩形的四边形)，然后计算两条对边的平均角度
    :param text_box:
    :return:
    """
    # 取长边
    w_sum = abs(text_box[1][0] - text_box[0][0]) + abs(text_box[2][0] - text_box[3][0])
    h_sum = abs(text_box[3][1] - text_box[0][1]) + abs(text_box[2][1] - text_box[1][1])
    if w_sum > h_sum:
        angle = (calculate_line_angle(text_box[0], text_box[1]) + calculate_line_angle(text_box[3], text_box[2])) / 2
    else:
        angle = (calculate_line_angle(text_box[0], text_box[3]) + calculate_line_angle(text_box[1], text_box[2])) / 2
    return angle

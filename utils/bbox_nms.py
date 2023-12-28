import numpy as np


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes: np.ndarray, scores: np.ndarray, score_threshold: float, iou_threshold: float) -> list:
    """
    对批量的目标检测结果进行多类别分别进行nms计算
    boxes: shape为(batch_size, obj_count, point_count)
    scores: shape为(batch_size, class_count, obj_count)
    score_threshold: box预测得分阈值
    iou_threshold:nms中iou阈值
    return: [[标签class_id，得分scores，x_min，y_min，x_max，y_max]]
    """
    batch_size = boxes.shape[0]
    rets = []
    # 依次处理各个批次数据
    for i in range(batch_size):
        # 单个批次结果multiclass_nms
        bboxes_i = boxes[i]
        scores_i = np.max(scores[i], axis=0)
        # box置信度阈值过滤
        bboxes_i = bboxes_i[scores_i > score_threshold]
        class_ids = np.argmax(scores[i][:, scores_i > score_threshold], axis=0)
        scores_i = scores_i[scores_i > score_threshold]
        unique_class_ids = np.unique(class_ids)
        # 遍历所有类别，进行单分类NMS
        one_batch_keep = []
        for class_id in unique_class_ids:
            # 获取单个类别的boxes、scores
            class_indices = np.where(class_ids == class_id)[0]
            class_boxes = bboxes_i[class_indices, :]
            class_scores = scores_i[class_indices]
            # nms过滤
            keep_indices = nms(class_boxes, class_scores, iou_threshold)
            # boxes信息组装 [标签class_id，得分scores，x_min，y_min，x_max，y_max]
            keep_bboxes = bboxes_i[keep_indices]
            keep_scores = class_scores[keep_indices]
            keep_results = np.zeros([keep_scores.shape[0], 6])
            keep_results[:, 0] = class_id
            keep_results[:, 1] = keep_scores[:]
            keep_results[:, 2:6] = keep_bboxes[:, :]
            one_batch_keep.extend(keep_results)
        rets.append(one_batch_keep)
    rets = np.array(rets)
    return rets


def compute_iou(box, boxes):
    # Compute x_min, y_min, x_max, y_max for both boxes
    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

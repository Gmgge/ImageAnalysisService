import os
from libs.base.analysis_args import AnalysisArgs
from conf.service_args import project_root


def init_rec_args():
    args = AnalysisArgs()
    args.model_path = os.path.join(project_root, "data/seal/recognition")
    args.threshold = 0.6  # 置信度阈值，由于未进行负样本训练，该阈值较高
    args.max_len = 50  # 最长文本长度
    args.input_shape = (384, 384)  # 模式输入图像尺度
    return args


# 初始化参数,使用需要用import导入，防止不必要的二次复制 例如 import **.config.**_args as **_args
trocr_args = init_rec_args()


if __name__ == "__main__":
    print(trocr_args)

import os
from libs.base.analysis_args import AnalysisArgs


def init_args():
    args = AnalysisArgs()
    return args


# 初始化参数,使用需要用import导入，防止不必要的二次复制 例如 import **.config.**_args as **_args
seal_rec_args = init_args()


if __name__ == "__main__":
    print(seal_rec_args)

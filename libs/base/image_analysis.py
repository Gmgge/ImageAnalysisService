from argparse import Namespace
from abc import abstractmethod
from utils.log_init import logger


class BaseImageAnalysis(object):
    """
    图像分析基础类，用于支持pipeline分析功能
    """

    def __init__(self, module_args: Namespace = None) -> None:
        pass

    @abstractmethod
    def analysis(self, **kwargs):
        """
        分析图像数据
        """


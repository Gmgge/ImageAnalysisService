import os
import sys
import conf.global_variable as global_variable
from time import time
from typing import List
from libs.base.image_analysis import BaseImageAnalysis
from conf.service_args import open_models
from utils.log_init import logger
from utils.web_utils import add_time_info
from utils.web_exception import AnalysisException
from utils.file_utils import check_dirs_in_path


class AnalysisPipeline(object):
    """
    图像分析pipeline模块
    """

    def __init__(self) -> None:
        self.analysis_modules = self.init_analysis_modules()

    @staticmethod
    def init_analysis_modules():
        """
        使用分析基类获取其子类，之后依次初始化，并传入一份引用到全局变量中
        """
        modules_dict = {}
        for analysis_class in BaseImageAnalysis.__subclasses__():
            class_file_path = os.path.abspath(sys.modules[analysis_class.__module__].__file__)
            module_name = check_dirs_in_path(open_models, class_file_path)
            if module_name:
                init_one_class = analysis_class()
                modules_dict[module_name] = init_one_class
                # 将初始化后的模块指针传递一份给全局变量_global_module_pool,用于有依赖的模块从中调用其他模块
                global_variable.module_pools[module_name] = init_one_class
        return modules_dict

    @add_time_info
    def analysis_image(self, image, tasks: List[str], tasks_args: dict = {}):
        """
        根据任务列表，进行分析pipeline任务编排，最后返回分析结果
        return {task_0: task_0_res}, 某个任务失败，默认分析结果为None
        """
        analysis_res = dict()
        for task_name in tasks:  # 任务名task_name对应分析模块analysis_module
            if task_name in self.analysis_modules.keys():
                # 根据任务名调用相应分析模块进行分析
                start_time = time()
                current_module = self.analysis_modules.get(task_name, None)
                current_args = tasks_args.get(task_name, {})
                current_args["image"] = image
                # 调度分析模块
                try:
                    current_res = current_module.analysis(**current_args)
                # 自定义异常抛出，需要处理
                except AnalysisException as e:
                    raise e
                except Exception as e:
                    logger.exception(e)
                    logger.error("任务：{}分析失败，耗时：{}s".format(task_name, round(time() - start_time, 2)))
                    current_res = None
                    # 整合分析结果
                    logger.info("任务：{}分析结束，耗时：{}s".format(task_name, round(time() - start_time, 2)))
            else:
                logger.error("请求中存在任务：{}，当前分析端未支持".format(task_name))
                current_res = None
            analysis_res[task_name] = current_res
        return analysis_res


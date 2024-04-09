import conf.global_variable as global_variable
from fastapi import Form, File, UploadFile
from pydantic import BaseModel
from fastapi import APIRouter, Query
from typing import List
from utils.log_init import logger
from conf.service_args import service_config
from utils.image_utils import read_image_file
from utils.web_exception import openapi_response
from utils.web_utils import BaseResponseItem


# 构建本地文件请求体参数
class PipelineItem(BaseModel):
    image: str
    tasks: List[str] = Query(
        description="Candidate is a subset of:" + str(list(service_config["ModuleSwitch"].keys())))
    tasks_args: dict = {}


# 返回体信息
class ResponseItem(BaseResponseItem):
    data: dict


description = """
data: dict, 其元素为
- **task_name**: **task_res**
- **task_res** 未成功分析时，默认为None
"""

# 构建路由
router = APIRouter(prefix=f'/{service_config["ServiceName"]}', tags=['addition'])


@router.post("/online_analysis_pipeline",
             responses=openapi_response,
             response_model=ResponseItem,
             response_description=description)
def online_image_analysis(image: UploadFile = File(), tasks: str = Form(), tasks_args: str = Form()):
    """
    在线图像分析接口 form-data形式
    image 示例值：二进制文件 form-data形式
    tasks 示例值：["ocr"] 待分析任务列表，必须是["ocr", "seal_rec"]的子集
    tasks_args 示例值 {“ocr”：{...}} 待分析任务对应的参数 预留字段，非必要
    """
    logger.info(f"pipeline接收到multipart/form-data请求 image：{image.filename}， tasks：{eval(tasks)}，"
                f"tasks_args：{eval(tasks_args)}")
    image_data = read_image_file(image)
    return global_variable.image_analysis_pipeline.analysis_image(image_data,
                                                                  eval(tasks),
                                                                  eval(tasks_args))


@router.post("/analysis_pipeline",
             responses=openapi_response,
             response_model=ResponseItem,
             response_description=description)
def image_analysis(req_item: PipelineItem):
    """
    本地图像分析接口 application/json形式
    image 示例值：文件路径
    tasks 示例值：["ocr"] 待分析任务列表，必须是["ocr", "seal_rec"]的子集
    tasks_args 示例值 {“ocr”：{...}} 待分析任务对应的参数 预留字段，非必要
    """
    logger.info(f"pipeline接收到application/json请求 image：{req_item.image}， tasks：{req_item.tasks}，"
                f"tasks_args：{req_item.tasks_args}")
    image_data = read_image_file(req_item.image)
    return global_variable.image_analysis_pipeline.analysis_image(image_data, req_item.tasks, req_item.tasks_args)

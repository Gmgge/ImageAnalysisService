import conf.global_variable as global_variable
from fastapi import Form, File, UploadFile
from pydantic import BaseModel
from fastapi import APIRouter, Query
from typing import List
from utils.log_init import logger
from core.analysis_pipeline.analysis_pipeline_args import task_candidate_list
from utils.image_utils import read_image_file
from utils.web_exception import openapi_response
from utils.web_utils import BaseResponseItem


# 构建本地文件请求体参数
class PipelineItem(BaseModel):
    image: str
    tasks: List[str] = Query(
        description="Candidate is a subset of:" + str(task_candidate_list))
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
router = APIRouter(prefix='/image_analysis', tags=['addition'])


@router.post("/analysis_pipeline",
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
    logger.info("pipeline接收到multipart/form-data请求 image：{}， tasks：{}，tasks_args：{}".format(image.filename,
                                                                                           eval(tasks),
                                                                                           eval(tasks_args)))
    image_data = read_image_file(image)
    return global_variable.image_analysis_pipeline.analysis_image(image_data,
                                                                  eval(tasks),
                                                                  eval(tasks_args))

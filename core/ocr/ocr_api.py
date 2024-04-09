import conf.global_variable as global_variable
from pydantic import BaseModel
from fastapi import APIRouter
from utils.log_init import logger
from utils.image_utils import read_image_file
from utils.web_exception import openapi_response


# 构建本地文件请求体参数
class PipelineItem(BaseModel):
    imgstring: str
    mode: str


# 返回体信息
class ResponseItem(BaseModel):
    data: dict
    message: str
    status: int
    timestamp: str


description = """
data: dict, 其元素为
- **content**: ocr识别内容
"""

# 构建路由
router = APIRouter(prefix='/image_analysis', tags=['addition'])


@router.post("/ocr",
             responses=openapi_response,
             response_model=ResponseItem,
             response_description=description)
def ocr(req_item: PipelineItem):
    """
    本地图像分析接口 application/json形式
    image 示例值：文件路径
    tasks 示例值：["ocr"] 待分析任务列表，必须是["ocr", "seal_rec"]的子集
    tasks_args 示例值 {“ocr”：{...}} 待分析任务对应的参数 预留字段，非必要
    """
    logger.info(f"ocr 接收到 application/json 请求 imgstring：{req_item.imgstring}， mode：{req_item.mode}")
    image_data = read_image_file(req_item.imgstring)
    analysis_res = global_variable.image_analysis_pipeline.analysis_image(image_data, ["ocr"])
    # 兼容旧接口
    ocr_res = {"data": {"content": analysis_res["data"]["ocr"],
                        "cost": analysis_res["cost"],
                        "filename": req_item.imgstring,
                        "text": []},
               "message": analysis_res["message"],
               "status": analysis_res["status"],
               "timestamp": analysis_res["timestamp"]}

    return ocr_res

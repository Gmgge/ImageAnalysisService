import conf.global_variable as global_variable
from pydantic import BaseModel, RootModel
from typing import List
from fastapi import APIRouter
from utils.log_init import logger
from utils.image_utils import read_image_file
from utils.web_exception import openapi_response
from conf.service_args import service_config
from conf.global_constant import IMG_PATH, IMG_STREAM


# 构建本地文件请求体参数
class PipelineItem(BaseModel):
    imgstring: str
    mode: str


# 返回体信息
class OneResponseItem(BaseModel):
    data: dict
    message: str
    status: int
    timestamp: str


ResponseItem = RootModel[List[OneResponseItem]]

description = """
data: dict, 其元素为
- **content**: ocr识别内容
"""

# 构建路由
router = APIRouter(prefix=f'/{service_config["ServiceName"]}', tags=['addition'])


@router.post("/ocr",
             responses=openapi_response,
             response_model=ResponseItem,
             response_description=description)
def ocr(req_item: PipelineItem):
    """
    本地图像分析接口 application/json形式
    imgstring:文件路径，文件base64编码
    mode:文件传输形式 值域 [imgpath:本地文件路径, imgstream:文件base64编码]
    """
    if req_item.mode == IMG_STREAM:
        img_info = req_item.imgstring[:20]
    elif req_item.mode == IMG_PATH:
        img_info = req_item.imgstring
    logger.info(f"ocr 接收到 application/json 请求 imgstring：{img_info}， mode：{req_item.mode}")
    image_data = read_image_file(req_item.imgstring, req_item.mode)
    analysis_res = global_variable.image_analysis_pipeline.analysis_image(image_data, ["ocr"])
    # 兼容旧接口
    ocr_res = {"data": {"content": analysis_res["data"]["ocr"],
                        "cost": analysis_res["cost"],
                        "filename": req_item.imgstring,
                        "text": []},
               "message": analysis_res["message"],
               "status": analysis_res["status"],
               "timestamp": analysis_res["timestamp"]}

    return [ocr_res]

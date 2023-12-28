import uvicorn
from fastapi import FastAPI, Request
import conf.global_variable as global_variable
from core.analysis_pipeline.analysis_pipeline import AnalysisPipeline
from utils.log_init import logger
from utils.web_utils import is_port_usable
from conf.service_args import service_config, router_file_root
from utils.web_utils import auto_include_router
from web.exception_handlers import add_exception_handlers


def creat_image_analysis_app(create_pipeline_sign: bool = True):
    """

    """
    # 检查端口
    assert is_port_usable(service_config["Web"]["port"]), "ImageAnalysisService分析服务端口被占用"
    # 构建web api
    logger.info("开始构建web app")
    app = FastAPI()

    # 构建分析模块 为兼容gunicorn下GPU调度，需要在post_fork中初始化模型
    if create_pipeline_sign:
        logger.info("初始分析模块")
        global_variable.image_analysis_pipeline = AnalysisPipeline()
        logger.info("分析模块初始化成功")

    # 按照规则自动注册路由，如果需要自动注册，请按照readme中的规范定义路由api文件
    logger.info("开始注册web路由模块")
    auto_include_router(app, router_file_root)

    # 重载异常情况处理器
    add_exception_handlers(app)

    logger.info("web app构建成功")
    return app


if __name__ == "__main__":
    image_analysis_app = creat_image_analysis_app()
    uvicorn.run(image_analysis_app, host=service_config["Web"]["ip"], port=service_config["Web"]["port"])

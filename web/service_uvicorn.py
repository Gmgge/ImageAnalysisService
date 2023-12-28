from web.service_app import creat_image_analysis_app
from conf.service_args import service_config
from utils.log_init import logger, push_uvicorn_log


def uvicorn_sever_run():
    """
    使用uvicorn作为服务器启动app服务
    """
    import uvicorn
    logger.info("当前使用web服务器：uvicorn")
    # 构建image_analysis app
    image_analysis_app = creat_image_analysis_app()
    config = uvicorn.Config(image_analysis_app, host=service_config["Web"]["ip"], port=service_config["Web"]["port"],
                            access_log=True)
    server = uvicorn.Server(config)
    # 将uvicorn push 到统一日志中
    push_uvicorn_log()
    server.run()


if __name__ == "__main__":
    uvicorn_sever_run()

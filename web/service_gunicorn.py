# 用于启动gunicorn的配置文件，可以设置相应的hook，用于轻量级多GPU调度
import gunicorn.app.base
import conf.global_variable as global_variable
from utils.log_init import logger, push_gunicorn_log
from conf.service_args import service_config
from web.service_app import creat_image_analysis_app
from core.analysis_pipeline.analysis_pipeline import AnalysisPipeline


def on_starting(sever):
    """
    主进程初始化之前调用
    """
    pass


def post_fork(server, worker):
    """
    Called just after a worker has been forked.
    """
    logger.info("进程：{} fork成功， hook开始拉起模型初始化".format(worker.pid))
    global_variable.image_analysis_pipeline = AnalysisPipeline()


image_analysis_options = {
        "bind": "{}:{}".format(service_config["Web"]["ip"], service_config["Web"]["port"]),  # 绑定的地址与端口号
        "workers": service_config["Web"]["workers"],  # 多进程workers数量
        "worker_class": "uvicorn.workers.UvicornWorker",
        "workers_connections": 1000,  # 最大并发量
        "graceful_timeout": service_config["Web"]["timeout"],  # 超时时间
        "timeout": service_config["Web"]["timeout"],
        "reload": False,
        "post_fork": post_fork

    }


class StandaloneApplication(gunicorn.app.base.BaseApplication):

    def init(self, parser, opts, args):
        pass

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                  if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def gunicorn_sever_run():
    logger.info("当前使用web服务器：gunicorn")
    # 使用gunicorn多进程进行管理，模型在fork之后初始化，以适配GPU下多进程情况
    image_analysis_app = creat_image_analysis_app(create_pipeline_sign=False)
    # gunicorn push 到统一日志中
    push_gunicorn_log()
    StandaloneApplication(image_analysis_app, image_analysis_options).run()


if __name__ == "__main__":
    gunicorn_sever_run()

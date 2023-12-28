import platform


def main():
    """
    image_analysis 启动主函数
    启动会根据机器操作系统选择不同的web服务器
    """
    # 非必要不导入相关运行组件
    if platform.system() == "Windows":
        from web.service_uvicorn import uvicorn_sever_run
        uvicorn_sever_run()

    else:
        from web.service_gunicorn import gunicorn_sever_run
        gunicorn_sever_run()


if __name__ == "__main__":
    main()

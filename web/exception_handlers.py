from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from utils.web_exception import AnalysisException


def add_exception_handlers(app):
    # 重载请求验证异常处理器
    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
        """
        请求参数验证异常
        :param request: 请求头信息
        :param exc: 异常对象
        """
        response_data = {"code": 422,
                         "message": str(exc.errors()),
                         "data": {},
                         "cost": 0.0}
        return JSONResponse(content=response_data, status_code=response_data["code"])

    # 重载分析异常处理器
    @app.exception_handler(AnalysisException)
    async def request_analysis_exception_handler(request: Request, exc: AnalysisException):
        """
        请求参数验证异常
        :param request: 请求头信息
        :param exc: 异常对象
        """
        response_data = {"code": exc.status_code,
                         "message": str(exc.detail),
                         "data": {},
                         "cost": 0.0}
        return JSONResponse(content=response_data, status_code=response_data["code"])


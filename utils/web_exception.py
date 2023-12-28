from typing import Any, Dict, Optional
from fastapi import HTTPException
from pydantic import BaseModel
from utils.log_init import logger


class AnalysisException(HTTPException):
    def __init__(self,
                 status_code: int,
                 detail: Any = None,
                 headers: Dict[str, str] = None) -> None:
        logger.error(detail),
        super().__init__(status_code, detail, headers)


class FileNotExistException(AnalysisException):
    def __init__(self, file_info: str = "", headers: Dict[str, str] = None) -> None:
        status_code = 400
        detail = "file: {} not exit ".format(file_info)
        super().__init__(status_code, detail, headers)


class UnsupportedFileTypeException(AnalysisException):
    def __init__(self, file_info: str = "", headers: Dict[str, str] = None) -> None:
        status_code = 415
        detail = "file: {} this type is not supported ".format(file_info)
        super().__init__(status_code, detail, headers)


class UnsupportedParametersException(AnalysisException):
    def __init__(self, error_info: str = "", headers: Dict[str, str] = None) -> None:
        status_code = 460
        detail = error_info
        super().__init__(status_code, detail, headers)


class AnalysisFailedException(AnalysisException):
    def __init__(self, file_info: str = "", headers: Dict[str, str] = None) -> None:
        status_code = 500
        detail = "file: {} analysis failed".format(file_info)
        super().__init__(status_code, detail, headers)


# todo 更优雅的状态码api描述
# 生成异常描述 完善自动化接口文档内容
class BaseResponseItem(BaseModel):
    cost: float
    code: int
    message: str
    data: str


openapi_response = {
    400: {"description": "File Not Exist Error", "model": BaseResponseItem},
    415: {"description": "Unsupported FileType Error", "model": BaseResponseItem},
    422: {"description": "Validation Error", "model": BaseResponseItem},
    460: {"description": "Unsupported Parameters Error", "model": BaseResponseItem},
    500: {"description": "Analysis Failed Error", "model": BaseResponseItem},
}

if __name__ == "__main__":
    print(openapi_response)

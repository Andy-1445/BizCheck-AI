from pydantic import BaseModel, ConfigDict


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorDetail(APIModel):
    code: str
    message: str
    retryable: bool = False


class ErrorResponse(APIModel):
    detail: ErrorDetail


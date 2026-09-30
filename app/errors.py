"""领域异常：解析、校验、计算、存储各层使用的可读错误类型。

所有异常都带中文可读消息，接口层负责把它们转成合适的 HTTP 响应；
批量处理中捕获 :class:`RecordError` 只标记单条记录，不让整批失败。
"""

from __future__ import annotations


class SeismicError(Exception):
    """本服务所有领域异常的基类。"""


class RecordError(SeismicError):
    """单条记录解析或数据本身的问题（空文件、非数值行、步长异常等）。

    ``record_id`` 可在批量场景下由调用方补填。
    """

    def __init__(self, message: str, *, record_id: str | None = None) -> None:
        super().__init__(message)
        self.record_id = record_id


class ValidationError(SeismicError):
    """作业参数校验失败（阻尼越界、周期非递增、谱表非法等）。"""


class IntegrationError(SeismicError):
    """数值积分相关错误，典型场景是线性加速度法失稳且策略为拒绝。"""


class JobError(SeismicError):
    """作业状态相关错误（不存在、状态不允许某操作等）。"""

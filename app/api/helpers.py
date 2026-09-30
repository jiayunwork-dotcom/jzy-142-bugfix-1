"""接口层共用工具：ID 生成、JSON 响应、错误编码、批量解析。"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from flask import jsonify

from ..errors import RecordError, SeismicError, ValidationError
from ..parsing import parse_record_text

MAX_RECORDS_PER_BATCH = 200
MAX_POINTS_PER_RECORD = 200_000


def record_id(name: str, acc_bytes: bytes, dt: float, unit: str | None) -> str:
    """记录 ID = rec_ + 内容哈希前 16 位（确定性，重复上传同文件同 ID）。"""

    h = hashlib.sha256()
    h.update(name.encode("utf-8", "replace"))
    h.update(acc_bytes)
    h.update(f"{dt!r}|{unit!r}".encode())
    return "rec_" + h.hexdigest()[:16]


def job_id() -> str:
    """作业 ID 含 UUID4，保证不同提交可区分；结果确定性由内容保证。"""

    return "job_" + uuid.uuid4().hex[:16]


def ok(payload: Any, status: int = 200):
    return jsonify(payload), status


def decode_bytes(raw: bytes) -> str:
    """文本记录容错解码：UTF-8 优先，退回 GB18030，再退回 latin-1。"""

    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def parse_uploaded_file(
    raw: bytes,
    filename: str,
    *,
    dt: float | None,
    unit: str | None,
    max_points: int = MAX_POINTS_PER_RECORD,
):
    """:return: (ParsedRecord, None) 或 (None, 可读错误字符串)。"""

    try:
        if not raw or not raw.strip():
            raise RecordError(f"记录 {filename} 为空文件或全是空白")
        text = decode_bytes(raw)
        return (
            parse_record_text(
                text, filename=filename, dt=dt, unit=unit, max_points=max_points
            ),
            None,
        )
    except SeismicError as exc:
        return None, str(exc)


def error_body(message: str, **extra) -> dict:
    body = {"error": message}
    body.update(extra)
    return body


def require_record_limit(n: int) -> None:
    if n > MAX_RECORDS_PER_BATCH:
        raise ValidationError(
            f"一批最多 {MAX_RECORDS_PER_BATCH} 条记录，本次收到 {n} 条"
        )

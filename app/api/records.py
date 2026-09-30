"""记录上传与查询接口。

两种提交方式
------------
1. ``multipart/form-data``：字段名 ``files`` 可重复，每个文件一条记录；
   表单字段 ``unit``（m/s2|g）整批生效，``dt`` 作为单列格式默认步长。
2. ``application/json``::

       {"unit": "g", "dt": 0.01,
        "records": [
          {"name": "RSN1.AT2", "content": "# ...\n0.01 0.002\n...",
           "unit": "m/s2", "dt": null},
          {"name": "single.txt", "content_b64": "...", "dt": 0.005}
        ]}

   单条可用 ``unit`` / ``dt`` 覆盖整批默认值。

单条解析失败不影响其他记录：HTTP 200，逐条给出 status/error；失败
记录同样入库（status='error'，不参与作业），便于事后排查。
"""

from __future__ import annotations

import base64
import hashlib

from flask import Blueprint, current_app, request

from ..errors import SeismicError, ValidationError
from ..parsing import normalize_unit
from .helpers import (
    MAX_POINTS_PER_RECORD,
    error_body,
    ok,
    parse_uploaded_file,
    record_id,
    require_record_limit,
)

bp = Blueprint("records", __name__)


@bp.get("/api/records")
def list_records():
    storage = current_app.extensions["storage"]
    return ok({"records": storage.list_records()})


@bp.get("/api/records/<rid>")
def get_record(rid: str):
    storage = current_app.extensions["storage"]
    rows = {r["id"]: r for r in storage.list_records()}
    if rid not in rows:
        return ok(error_body(f"记录 {rid} 不存在"), 404)
    return ok({"record": rows[rid]})


def _as_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"步长 dt 必须是数值，收到 {v!r}") from exc


def _short_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:12]


def _store_one(storage, filename: str, raw: bytes | str, unit, dt) -> dict:
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    parsed, perr = parse_uploaded_file(
        raw, filename, dt=dt, unit=unit, max_points=MAX_POINTS_PER_RECORD
    )
    if perr is not None:
        rid = "rec_bad" + _short_hash(filename + perr)
        storage.upsert_record(
            rid, name=filename, dt=float(dt) if dt else 1.0,
            unit_input=_safe_unit(unit), fmt="unknown", metadata={},
            acc=None, status="error", error=perr,
        )
        return {"name": filename, "record_id": rid, "status": "error",
                "error": perr}

    if parsed.unit is None:
        # 单位必须显式声明（m/s2 或 g）：量级搞错会导致谱值系统性偏差
        perr = (
            f"记录 {filename} 缺少加速度单位声明：请在上传参数或文件头中"
            "指明 unit=m/s2 或 g"
        )
        rid = "rec_bad" + _short_hash(filename + perr)
        storage.upsert_record(
            rid, name=filename, dt=parsed.dt, unit_input=None,
            fmt=parsed.fmt, metadata=parsed.metadata, acc=None,
            status="error", error=perr,
        )
        return {"name": filename, "record_id": rid, "status": "error",
                "error": perr}

    rid = record_id(filename, parsed.acc.tobytes(), parsed.dt,
                    parsed.unit or "unknown")
    storage.upsert_record(
        rid,
        name=filename,
        dt=parsed.dt,
        unit_input=parsed.unit,
        fmt=parsed.fmt,
        metadata=parsed.metadata,
        acc=parsed.acc,
        status="ready",
    )
    return {
        "name": filename,
        "record_id": rid,
        "status": "ready",
        "dt": parsed.dt,
        "npts": parsed.npts,
        "duration": parsed.duration,
        "unit": parsed.unit,
        "fmt": parsed.fmt,
        "metadata": parsed.metadata,
    }


def _safe_unit(unit):
    try:
        return normalize_unit(unit)
    except SeismicError:
        return None


@bp.post("/api/records")
def upload_records():
    storage = current_app.extensions["storage"]
    ctype = (request.content_type or "")

    try:
        if "application/json" in ctype:
            payload = request.get_json(force=True, silent=True)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("records"), list
            ):
                raise ValidationError(
                    'JSON 请求体需为 {"records": [...]}，每项含 name 与 '
                    "content 或 content_b64"
                )
            default_unit = payload.get("unit")
            default_dt = _as_float(payload.get("dt"))
            files = []
            for i, e in enumerate(payload["records"]):
                if not isinstance(e, dict) or "name" not in e or (
                    "content" not in e and "content_b64" not in e
                ):
                    raise ValidationError(
                        f"第 {i + 1} 条记录缺少 name 或 content/content_b64"
                    )
                if "content" in e:
                    raw = str(e["content"]).encode("utf-8")
                else:
                    try:
                        raw = base64.b64decode(e["content_b64"], validate=True)
                    except Exception as exc:  # noqa: BLE001
                        raise ValidationError(
                            f"记录 {e['name']} 的 content_b64 无法 base64 解码：{exc}"
                        ) from exc
                files.append((
                    e["name"], raw,
                    e.get("unit", default_unit),
                    _as_float(e.get("dt")) if e.get("dt") is not None else default_dt,
                ))
        else:
            default_unit = request.form.get("unit")
            default_dt = _as_float(request.form.get("dt"))
            files = [
                (f.filename or "unnamed", f.read(), default_unit, default_dt)
                for f in request.files.getlist("files")
            ]
            if not files:
                return ok(
                    error_body(
                        "没有收到任何记录：请以 multipart 上传 files 字段，或发送 "
                        'Content-Type: application/json 的 {"records": [...]}'
                    ),
                    400,
                )
        require_record_limit(len(files))
    except ValidationError as exc:
        return ok(error_body(str(exc)), 400)

    out = [_store_one(storage, name, raw, unit, dt)
           for name, raw, unit, dt in files]
    n_ready = sum(1 for x in out if x["status"] == "ready")
    return ok({
        "uploaded": len(out),
        "ready": n_ready,
        "failed": len(out) - n_ready,
        "records": out,
    })

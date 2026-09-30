"""作业创建、查询、取消接口。

POST /api/jobs/spectrum  {"record_ids": [...], "periods"?, "dampings"?,
                          "method"?, "instability_policy"?, "baseline_mode"?}
POST /api/jobs/match     额外需要 design_spectrum{t1,t2,...}、t1/t2、
                          target_damping、s_min/s_max、top_n、quantity、
                          bounds_policy
GET  /api/jobs                 最近作业列表（不含大结果体）
GET  /api/jobs/<id>            作业状态 + 结果（完成后）
GET  /api/jobs/<id>/items      逐条作业项（执行中可查部分结果）
POST /api/jobs/<id>/cancel     取消排队/执行中的作业
"""

from __future__ import annotations

from flask import Blueprint, current_app

from ..errors import SeismicError, ValidationError
from ..spectrum import prepare_spectrum_request
from .helpers import error_body, job_id, ok, require_record_limit

bp = Blueprint("jobs", __name__)

# 两个创建端点共用、且会立即校验的谱计算参数
_SPECTRUM_KEYS = (
    "periods", "dampings", "method", "instability_policy", "baseline_mode",
)


def _payload():
    from flask import request

    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    return data


def _resolve_record_ids(storage, record_ids):
    if not isinstance(record_ids, list) or not record_ids:
        raise ValidationError("record_ids 必须是非空数组")
    require_record_limit(len(record_ids))
    seen, ordered = set(), []
    for x in record_ids:
        if not isinstance(x, str) or not x:
            raise ValidationError(f"record_ids 中存在非法项：{x!r}")
        if x not in seen:
            seen.add(x)
            ordered.append(x)
    if len(ordered) != len(record_ids):
        raise ValidationError("record_ids 中存在重复项，请去重后再提交")
    missing = [rid for rid in ordered if storage.get_record(rid) is None]
    if missing:
        raise ValidationError(f"以下记录不存在：{', '.join(missing)}")
    not_ready = []
    for rid in ordered:
        rec = storage.get_record(rid)
        if rec.status != "ready":
            not_ready.append(f"{rec.name}({rid}): {rec.error or rec.status}")
    if not_ready:
        raise ValidationError(
            "以下记录不可用（解析失败），请先修正后重新上传：" + "；".join(not_ready)
        )
    return ordered


def _spectrum_params(data) -> dict:
    # 立即触发参数校验，错误在创建时返回而不是运行中
    req = prepare_spectrum_request(
        periods=data.get("periods"),
        dampings=data.get("dampings", 0.05),
        method=data.get("method", "average_acceleration"),
        instability_policy=data.get("instability_policy", "refine"),
        baseline_mode=data.get("baseline_mode", "mean"),
    )
    return {
        "periods": req.periods.tolist(),
        "dampings": list(req.dampings),
        "method": req.method,
        "instability_policy": req.instability_policy,
        "baseline_mode": req.baseline_mode,
    }


def _create_job(job_type: str, params: dict, record_ids: list[str]):
    storage = current_app.extensions["storage"]
    scheduler = current_app.extensions["scheduler"]
    jid = job_id()
    storage.create_job(jid, job_type, params, total=len(record_ids))
    storage.add_job_items(jid, [(rid, seq) for seq, rid in enumerate(record_ids)])
    scheduler.notify()
    return ok({
        "job_id": jid,
        "type": job_type,
        "status": "queued",
        "total": len(record_ids),
    }, 201)


@bp.post("/api/jobs/spectrum")
def create_spectrum_job():
    storage = current_app.extensions["storage"]
    try:
        data = _payload()
        record_ids = _resolve_record_ids(storage, data.get("record_ids"))
        params = _spectrum_params(data)
    except SeismicError as exc:
        return ok(error_body(str(exc)), 400)
    return _create_job("spectrum", params, record_ids)


@bp.post("/api/jobs/match")
def create_match_job():
    storage = current_app.extensions["storage"]
    try:
        data = _payload()
        record_ids = _resolve_record_ids(storage, data.get("record_ids"))
        params = _spectrum_params(data)

        ds = data.get("design_spectrum")
        if not isinstance(ds, dict):
            raise ValidationError("缺少 design_spectrum：{periods, values, unit}")
        from ..matching import make_design_spectrum

        design = make_design_spectrum(
            ds.get("periods"), ds.get("values"), ds.get("unit", "g")
        )
        try:
            t1 = float(data["t1"])
            t2 = float(data["t2"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("必须给出数值型匹配区间 t1、t2（秒）") from exc
        if not (0 < t1 < t2):
            raise ValidationError(f"匹配区间非法：需 0 < t1 < t2，收到 {t1}, {t2}")
        if t1 < design.periods[0] or t2 > design.periods[-1]:
            raise ValidationError(
                f"匹配区间 [{t1}, {t2}] 超出设计谱范围 "
                f"[{design.periods[0]}, {design.periods[-1]}]"
            )
        target_damping = data.get("target_damping", 0.05)
        try:
            td = float(target_damping)
        except (TypeError, ValueError) as exc:
            raise ValidationError("target_damping 必须是数值") from exc
        if not (0.0 <= td <= 1.0):
            raise ValidationError(f"target_damping 必须在 0 到 1 之间，收到 {td}")

        s_min = float(data.get("s_min", 0.5))
        s_max = float(data.get("s_max", 3.0))
        if not (0 < s_min <= s_max):
            raise ValidationError(
                f"缩放系数上下限非法：需 0 < s_min <= s_max，收到 {s_min}, {s_max}"
            )
        top_n = int(data.get("top_n", min(7, len(record_ids))))
        if top_n < 1:
            raise ValidationError("top_n 至少为 1")
        quantity = data.get("quantity", "psa")
        if quantity not in ("psa", "sa"):
            raise ValidationError("quantity 只支持 psa 或 sa")
        bounds_policy = data.get("bounds_policy", "clamp")
        if bounds_policy not in ("clamp", "reject"):
            raise ValidationError("bounds_policy 只支持 clamp 或 reject")
        if ds.get("unit", "g") not in ("g", "m/s2"):
            raise ValidationError("设计谱单位只支持 g 或 m/s2")

        params.update({
            "design_spectrum": {
                "periods": design.periods.tolist(),
                "values": design.values.tolist(),
                "unit": design.unit,
            },
            "t1": t1,
            "t2": t2,
            "target_damping": td,
            "s_min": s_min,
            "s_max": s_max,
            "top_n": top_n,
            "quantity": quantity,
            "bounds_policy": bounds_policy,
        })
    except SeismicError as exc:
        return ok(error_body(str(exc)), 400)
    return _create_job("match", params, record_ids)


@bp.get("/api/jobs")
def list_jobs():
    storage = current_app.extensions["storage"]
    return ok({"jobs": storage.list_jobs()})


@bp.get("/api/jobs/<jid>")
def get_job(jid: str):
    storage = current_app.extensions["storage"]
    job = storage.get_job(jid)
    if job is None:
        return ok(error_body(f"作业 {jid} 不存在"), 404)
    return ok({"job": job})


@bp.get("/api/jobs/<jid>/items")
def get_job_items(jid: str):
    storage = current_app.extensions["storage"]
    if storage.get_job(jid) is None:
        return ok(error_body(f"作业 {jid} 不存在"), 404)
    items = storage.list_job_items(jid)
    counts = storage.counts_by_status(jid)
    return ok({"job_id": jid, "counts": counts, "items": items})


@bp.post("/api/jobs/<jid>/cancel")
def cancel_job(jid: str):
    storage = current_app.extensions["storage"]
    if storage.get_job(jid) is None:
        return ok(error_body(f"作业 {jid} 不存在"), 404)
    changed = storage.request_cancel(jid)
    if not changed:
        return ok(
            error_body("作业已结束（完成/失败/取消/中断），无法取消"),
            409,
        )
    scheduler = current_app.extensions["scheduler"]
    scheduler.notify()
    return ok({"job_id": jid, "status": "cancelled"})

"""作业执行逻辑：反应谱作业与匹配作业的逐记录处理。

单条记录失败（数据库中状态异常、数据损坏、积分被 reject 等）只标记
对应作业项为 error/skipped，绝不拖垮整批；作业整体完成后 result 中
同时给出汇总与逐条结果（匹配作业还给出排序与平均谱）。
"""

from __future__ import annotations

from .errors import SeismicError
from .matching import make_design_spectrum, match_batch
from .parsing import GRAVITY
from .spectrum import compute_spectrum, prepare_spectrum_request
from .storage import Storage


def _load_ready_record(storage: Storage, record_id: str):
    rec = storage.get_record(record_id)
    if rec is None:
        return None, f"记录 {record_id} 不存在，可能已被删除"
    if rec.status != "ready" or rec.acc.size == 0:
        return None, f"记录 {rec.name}({record_id}) 不可用：{rec.error or rec.status}"
    return rec, None


def run_spectrum_job(job_id: str, storage: Storage, is_cancelled) -> dict:
    """执行一条 spectrum 作业；``is_cancelled()`` 为取消轮询回调。"""

    params = storage.get_job_params(job_id)
    req = prepare_spectrum_request(
        periods=params.get("periods"),
        dampings=params.get("dampings", 0.05),
        method=params.get("method", "average_acceleration"),
        instability_policy=params.get("instability_policy", "refine"),
        baseline_mode=params.get("baseline_mode", "mean"),
    )
    items = storage.list_job_items(job_id)
    records_out = []
    errors = 0
    skipped = 0

    for i, item in enumerate(items):
        if is_cancelled():
            storage.finish_cancelled(job_id)
            return {"cancelled": True}
        storage.set_item_running(job_id, item["seq"])
        rid = item["record_id"]
        rec, load_err = _load_ready_record(storage, rid) if rid else (None, "缺少记录 id")
        if load_err:
            storage.set_item_skipped(job_id, item["seq"], load_err)
            records_out.append({
                "seq": item["seq"], "record_id": rid,
                "status": "skipped", "error": load_err,
            })
            skipped += 1
            storage.set_job_progress(job_id, i + 1)
            continue
        try:
            factor = GRAVITY if rec.unit_input == "g" else 1.0
            result = compute_spectrum(rec.acc * factor, rec.dt, req)
            summary = {
                "seq": item["seq"],
                "record_id": rid,
                "name": rec.name,
                "status": "done",
                "npts": rec.npts,
                "dt": rec.dt,
                "ground_motion": result["ground_motion"],
                "baseline": result["baseline"],
                "free_vibration": result["free_vibration"],
                "units": result["units"],
                "spectra": result["spectra"],
            }
            storage.set_item_result(job_id, item["seq"], summary)
            records_out.append(summary)
        except SeismicError as exc:
            msg = f"记录 {rec.name}({rid}) 计算失败：{exc}"
            storage.set_item_error(job_id, item["seq"], msg)
            records_out.append({
                "seq": item["seq"], "record_id": rid, "name": rec.name,
                "status": "error", "error": str(exc),
            })
            errors += 1
        storage.set_job_progress(job_id, i + 1)

    return {
        "cancelled": False,
        "config": {
            "periods": req.periods.tolist(),
            "dampings": list(req.dampings),
            "method": req.method,
            "instability_policy": req.instability_policy,
            "baseline_mode": req.baseline_mode,
        },
        "records": records_out,
        "summary": {
            "total": len(items),
            "done": len(items) - errors - skipped,
            "error": errors,
            "skipped": skipped,
        },
    }


def run_match_job(job_id: str, storage: Storage, is_cancelled) -> dict:
    """执行一条 match 作业：先逐条算谱，再做整体匹配筛选。"""

    params = storage.get_job_params(job_id)
    req = prepare_spectrum_request(
        periods=params.get("periods"),
        dampings=params.get("dampings", 0.05),
        method=params.get("method", "average_acceleration"),
        instability_policy=params.get("instability_policy", "refine"),
        baseline_mode=params.get("baseline_mode", "mean"),
    )
    target_damping = float(params["target_damping"])
    if target_damping not in req.dampings:
        # 请求阻尼列表里没有目标阻尼时补算
        req.dampings = tuple(sorted(set(req.dampings) | {target_damping}))

    dp = params["design_spectrum"]
    design = make_design_spectrum(dp["periods"], dp["values"], dp.get("unit", "g"))

    items = storage.list_job_items(job_id)
    candidates = []          # 供匹配模块使用
    per_record = []          # 逐条结果（含错误）
    errors = 0
    skipped = 0

    for i, item in enumerate(items):
        if is_cancelled():
            storage.finish_cancelled(job_id)
            return {"cancelled": True}
        storage.set_item_running(job_id, item["seq"])
        rid = item["record_id"]
        rec, load_err = _load_ready_record(storage, rid) if rid else (None, "缺少记录 id")
        if load_err:
            storage.set_item_skipped(job_id, item["seq"], load_err)
            per_record.append({
                "seq": item["seq"], "record_id": rid,
                "status": "skipped", "error": load_err,
            })
            skipped += 1
            storage.set_job_progress(job_id, i + 1)
            continue
        try:
            factor = GRAVITY if rec.unit_input == "g" else 1.0
            spec = compute_spectrum(rec.acc * factor, rec.dt, req)
            chosen = next(
                s for s in spec["spectra"]
                if abs(s["damping"] - target_damping) < 1e-12
            )
            quantity = params.get("quantity", "psa")
            values = chosen[quantity]
            # 设计谱若以 g 给出，记录谱同样换算到 g
            if design.unit == "g":
                values = [v / GRAVITY for v in values]
            candidates.append({
                "record_id": rid,
                "name": rec.name,
                "periods": req.periods.tolist(),
                quantity: values,
            })
            entry = {
                "seq": item["seq"],
                "record_id": rid,
                "name": rec.name,
                "status": "done",
                "ground_motion": spec["ground_motion"],
                "spectrum": chosen,
            }
            storage.set_item_result(job_id, item["seq"], entry)
            per_record.append(entry)
        except SeismicError as exc:
            msg = f"记录 {rec.name}({rid}) 计算失败：{exc}"
            storage.set_item_error(job_id, item["seq"], msg)
            per_record.append({
                "seq": item["seq"], "record_id": rid, "name": rec.name,
                "status": "error", "error": str(exc),
            })
            errors += 1
        storage.set_job_progress(job_id, i + 1)

    matching = match_batch(
        candidates,
        design,
        t1=float(params["t1"]),
        t2=float(params["t2"]),
        s_min=float(params.get("s_min", 0.5)),
        s_max=float(params.get("s_max", 3.0)),
        top_n=int(params.get("top_n", 7)),
        quantity=params.get("quantity", "psa"),
        bounds_policy=params.get("bounds_policy", "clamp"),
    )

    return {
        "cancelled": False,
        "config": {
            "periods": req.periods.tolist(),
            "dampings": list(req.dampings),
            "method": req.method,
            "instability_policy": req.instability_policy,
            "baseline_mode": req.baseline_mode,
            "target_damping": target_damping,
        },
        "matching": matching,
        "records": per_record,
        "summary": {
            "total": len(items),
            "spectra_done": len(candidates),
            "error": errors,
            "skipped": skipped,
        },
    }

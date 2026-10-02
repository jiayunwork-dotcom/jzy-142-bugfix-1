"""作业执行逻辑：反应谱作业与匹配作业的逐记录处理。

单条记录失败（数据库中状态异常、数据损坏、积分被 reject 等）只标记
对应作业项为 error/skipped，绝不拖垮整批；作业整体完成后 result 中
同时给出汇总与逐条结果（匹配作业还给出排序与平均谱）。

取消协议
--------
执行线程**不负责决定**作业终态，只负责在各检查点轮询取消标志并尽快
中止：条目之间、单条记录的积分循环内（每 256 个时间步）、匹配阶段
前后都有检查点。无论在哪个检查点停下，收尾都统一走
:meth:`Storage.finalize_job`，由存储层在一条事务里仲裁最终是 cancelled
还是 completed，因此「取消已确认」与「作业完成」不可能互相翻盘。
正在计算的那条记录采取**尽快中止、结果丢弃**策略（详见 README §8）。
"""

from __future__ import annotations

from .errors import JobCancelled, SeismicError
from .matching import make_design_spectrum, match_batch
from .parsing import GRAVITY
from .spectrum import compute_spectrum, prepare_spectrum_request
from .storage import Storage

# 测试钩子：在排序匹配阶段开始前调用（生产中为 None）。名字带下划线，
# 仅供回归测试 monkeypatch，用来把取消稳定地打进「最后一条算完、匹配
# 未完成」这一窗口；正常路径行为完全不受影响。
_before_match_hook = None


def _load_ready_record(storage: Storage, record_id: str):
    rec = storage.get_record(record_id)
    if rec is None:
        return None, f"记录 {record_id} 不存在，可能已被删除"
    if rec.status != "ready" or rec.acc.size == 0:
        return None, f"记录 {rec.name}({record_id}) 不可用：{rec.error or rec.status}"
    return rec, None


def _prepare_req(params: dict):
    return prepare_spectrum_request(
        periods=params.get("periods"),
        dampings=params.get("dampings", 0.05),
        method=params.get("method", "average_acceleration"),
        instability_policy=params.get("instability_policy", "refine"),
        baseline_mode=params.get("baseline_mode", "mean"),
    )


def run_spectrum_job(job_id: str, storage: Storage, is_cancelled) -> dict:
    """执行一条 spectrum 作业；``is_cancelled()`` 为取消轮询回调。"""

    params = storage.get_job_params(job_id)
    req = _prepare_req(params)
    items = storage.list_job_items(job_id)
    records_out = []
    errors = 0
    skipped = 0

    for i, item in enumerate(items):
        if is_cancelled():
            break
        seq = item["seq"]
        rid = item["record_id"]
        storage.set_item_running(job_id, seq)
        rec, load_err = _load_ready_record(storage, rid) if rid else (None, "缺少记录 id")
        if load_err:
            storage.set_item_skipped(job_id, seq, load_err)
            records_out.append({
                "seq": seq, "record_id": rid,
                "status": "skipped", "error": load_err,
            })
            skipped += 1
            storage.set_job_progress(job_id, i + 1)
            continue
        try:
            factor = GRAVITY if rec.unit_input == "g" else 1.0
            result = compute_spectrum(
                rec.acc * factor, rec.dt, req, is_cancelled=is_cancelled
            )
            summary = {
                "seq": seq,
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
            storage.set_item_result(job_id, seq, summary)
            records_out.append(summary)
        except JobCancelled:
            storage.set_item_cancelled(job_id, seq)
            break
        except SeismicError as exc:
            msg = f"记录 {rec.name}({rid}) 计算失败：{exc}"
            storage.set_item_error(job_id, seq, msg)
            records_out.append({
                "seq": seq, "record_id": rid, "name": rec.name,
                "status": "error", "error": str(exc),
            })
            errors += 1
        storage.set_job_progress(job_id, i + 1)
    else:
        # 所有条目正常跑完：仍由 finalize_job 仲裁——收尾提交前若取消
        # 标志先落库，判 cancelled，绝不在此处直接宣称完成。
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

    # break 出来：要么条目间检查到取消，要么单条计算中确认了取消
    return {"cancelled": True}


def run_match_job(job_id: str, storage: Storage, is_cancelled) -> dict:
    """执行一条 match 作业：先逐条算谱，再做整体匹配筛选。"""

    params = storage.get_job_params(job_id)
    req = _prepare_req(params)
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
    cancelled_early = False

    for i, item in enumerate(items):
        if is_cancelled():
            cancelled_early = True
            break
        seq = item["seq"]
        rid = item["record_id"]
        storage.set_item_running(job_id, seq)
        rec, load_err = _load_ready_record(storage, rid) if rid else (None, "缺少记录 id")
        if load_err:
            storage.set_item_skipped(job_id, seq, load_err)
            per_record.append({
                "seq": seq, "record_id": rid,
                "status": "skipped", "error": load_err,
            })
            skipped += 1
            storage.set_job_progress(job_id, i + 1)
            continue
        try:
            factor = GRAVITY if rec.unit_input == "g" else 1.0
            spec = compute_spectrum(
                rec.acc * factor, rec.dt, req, is_cancelled=is_cancelled
            )
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
                "seq": seq,
                "record_id": rid,
                "name": rec.name,
                "status": "done",
                "ground_motion": spec["ground_motion"],
                "spectrum": chosen,
            }
            storage.set_item_result(job_id, seq, entry)
            per_record.append(entry)
        except JobCancelled:
            storage.set_item_cancelled(job_id, seq)
            cancelled_early = True
            break
        except SeismicError as exc:
            msg = f"记录 {rec.name}({rid}) 计算失败：{exc}"
            storage.set_item_error(job_id, seq, msg)
            per_record.append({
                "seq": seq, "record_id": rid, "name": rec.name,
                "status": "error", "error": str(exc),
            })
            errors += 1
        storage.set_job_progress(job_id, i + 1)

    if cancelled_early:
        return {"cancelled": True}

    # 全部谱已算完，进入最后的排序匹配阶段。前后各留一个取消检查点：
    # match_batch 本身是毫秒级纯 Python，不为它侵入内部；落在这个窗口里
    # 的取消由 finalize_job 的原子仲裁判 cancelled（结果整体丢弃）。
    if is_cancelled():
        return {"cancelled": True}
    if _before_match_hook is not None:
        _before_match_hook(job_id, storage, is_cancelled)
    if is_cancelled():
        return {"cancelled": True}

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

    # match_batch 返回后不再有耗时操作，留给取消的窗口可忽略；终态仍由
    # finalize_job 仲裁（极小概率下取消恰好在提交前落库 -> cancelled）。
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

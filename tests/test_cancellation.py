"""取消 / 中断 / 终态一致性的回归测试。

守住的契约（见 README「取消、中断与条目终态」一节）：

- 取消一旦答复成功（200），作业最终只能是 cancelled，不会被任何
  收尾动作翻盘；取消来得太晚（作业已完成）则如实回 409；
- 作业进入任何终态（completed/failed/cancelled/interrupted）后，
  作业项里不再有 pending/running，各状态计数之和等于 total；
- 未跑完的条目能区分「因取消没跑」(cancelled) 与「因进程中断没
  跑完」(interrupted)；取消/中断前已算完的条目结果重启后仍可读；
- 取消落在不同时机（第一条中、最后一条中、结果落库前一刻、匹配
  阶段）反复多次打，每次都核对作业状态、逐条状态、计数三者一致。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import app.jobs as jobs_mod
from app.scheduler import Scheduler
from app.storage import ITEM_TERMINAL_STATES, Storage
from tests.conftest import wait_for_status

REPO_ROOT = Path(__file__).resolve().parents[1]

TERMINAL_JOB_STATES = ("completed", "failed", "cancelled", "interrupted")


# ---------------------------------------------------------------- helpers

def _put(storage: Storage, rid: str, n: int = 40_000, dt: float = 0.005) -> str:
    """直接往库里放一条可用记录（单列、g），返回记录 id。"""

    t = np.arange(n) * dt
    acc = 0.3 * np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.sin(2 * np.pi * 7.0 * t)
    storage.upsert_record(
        rid, name=f"{rid}.txt", dt=dt, unit_input="g", fmt="single_column",
        metadata={}, acc=acc, status="ready",
    )
    return rid


def _spectrum_params() -> dict:
    return {
        "periods": np.logspace(np.log10(0.02), np.log10(6), 100).tolist(),
        "dampings": [0.05],
        "method": "average_acceleration",
        "instability_policy": "refine",
        "baseline_mode": "mean",
    }


def _match_payload(record_ids) -> dict:
    return {
        "record_ids": record_ids,
        "periods": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0],
        "dampings": [0.05],
        "target_damping": 0.05,
        "design_spectrum": {
            "periods": [0.05, 0.1, 0.5, 1.0, 3.0],
            "values": [0.30, 0.30, 0.30, 0.30, 0.30],
            "unit": "g",
        },
        "t1": 0.1, "t2": 2.0,
        "s_min": 0.2, "s_max": 5.0, "top_n": 2,
        "quantity": "psa", "bounds_policy": "reject",
    }


def _check_consistent(job: dict, items: list[dict], counts: dict) -> None:
    """作业状态、逐条状态、计数三者一致（终态不变式）。"""

    assert len(items) == job["total"], "条目数应等于作业 total"
    assert sum(counts.values()) == job["total"], "各状态计数之和应等于 total"
    actual = Counter(i["status"] for i in items)
    for status, n in counts.items():
        assert actual.get(status, 0) == n, f"counts[{status}] 与条目实际不符"
    if job["status"] in TERMINAL_JOB_STATES:
        assert counts["pending"] == 0 and counts["running"] == 0, (
            f"作业已终态({job['status']})，不应再有 pending/running 条目"
        )
        assert all(i["status"] in ITEM_TERMINAL_STATES for i in items)
    if job["status"] == "cancelled":
        assert job["result"] is None, "取消的作业不得带汇总结果"
    if job["status"] == "completed":
        assert job["result"] is not None


def _consistent_via_api(client, jid: str):
    job = client.get(f"/api/jobs/{jid}").get_json()["job"]
    body = client.get(f"/api/jobs/{jid}/items").get_json()
    _check_consistent(job, body["items"], body["counts"])
    return job, body["items"], body["counts"]


def _consistent_via_storage(storage: Storage, jid: str):
    job = storage.get_job(jid)
    items = storage.list_job_items(jid)
    counts = storage.counts_by_status(jid)
    _check_consistent(job, items, counts)
    return job, items, counts


def _wait_job_status(client, jid: str, targets, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{jid}").get_json()["job"]
        if job["status"] in targets:
            return job
        time.sleep(0.01)
    raise AssertionError(f"作业 {jid} 未在 {timeout}s 内进入 {targets}")


def _wait_item_status(client, jid: str, seq: int, targets,
                      timeout: float = 30.0) -> list[dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        items = client.get(f"/api/jobs/{jid}/items").get_json()["items"]
        if items[seq]["status"] in targets:
            return items
        time.sleep(0.01)
    raise AssertionError(f"作业 {jid} 条目 {seq} 未在 {timeout}s 内进入 {targets}")


def _assert_status_stable(client, jid: str, status: str, duration: float = 0.5):
    """在 duration 内反复核对作业状态不被翻盘。"""

    deadline = time.time() + duration
    while time.time() < deadline:
        got = client.get(f"/api/jobs/{jid}").get_json()["job"]["status"]
        assert got == status, f"作业状态从 {status} 被翻转为 {got}"
        time.sleep(0.02)


def _wait_storage_job(storage: Storage, jid: str, targets, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = storage.get_job(jid)
        if job["status"] in targets:
            return job
        time.sleep(0.02)
    raise AssertionError(f"作业 {jid} 未在 {timeout}s 内进入 {targets}")


# ------------------------------------------------------- 取消时机（反复打）

def test_cancel_mid_single_record_never_flips(app_worker):
    """单条记录、计算中取消：取消被确认后，作业永远是 cancelled。

    用户原始场景：取消返回 200、状态 cancelled，但记录算完后作业被
    收尾动作翻成 completed 并带完整结果。这里反复打 3 次。
    """

    client = app_worker.test_client()
    storage = app_worker.extensions["storage"]
    for round_ in range(3):
        rid = _put(storage, f"single{round_}", n=60_000)
        jid = client.post("/api/jobs/spectrum", json={
            "record_ids": [rid], "dampings": [0.02, 0.05, 0.1],
        }).get_json()["job_id"]

        _wait_item_status(client, jid, 0, ("running",))
        rv = client.post(f"/api/jobs/{jid}/cancel")
        assert rv.status_code == 200
        assert rv.get_json()["status"] == "cancelled"

        # 正在算的条目算完并保留结果（let-it-finish）
        items = _wait_item_status(client, jid, 0, ("done",))
        assert items[0]["result"] is not None
        # 记录算完后作业状态不得被翻盘
        _assert_status_stable(client, jid, "cancelled")

        job, items, counts = _consistent_via_api(client, jid)
        assert job["status"] == "cancelled"
        assert [i["status"] for i in items] == ["done"]
        assert counts["done"] == 1


def test_cancel_during_first_record_of_batch(app_worker):
    """多条记录、第一条计算中取消：第一条算完保留，其余为 cancelled。"""

    client = app_worker.test_client()
    storage = app_worker.extensions["storage"]
    for round_ in range(3):
        ids = [_put(storage, f"first{round_}_{i}", n=40_000) for i in range(3)]
        jid = client.post("/api/jobs/spectrum", json={
            "record_ids": ids, "dampings": [0.05],
        }).get_json()["job_id"]

        _wait_item_status(client, jid, 0, ("running",))
        rv = client.post(f"/api/jobs/{jid}/cancel")
        assert rv.status_code == 200

        _wait_item_status(client, jid, 0, ("done",))
        _assert_status_stable(client, jid, "cancelled")

        job, items, counts = _consistent_via_api(client, jid)
        assert job["status"] == "cancelled"
        assert [i["status"] for i in items] == ["done", "cancelled", "cancelled"]
        assert items[0]["result"] is not None, "取消前已算完的条目结果应保留"
        assert counts["done"] == 1 and counts["cancelled"] == 2


def test_cancel_during_last_record_of_batch(app_worker):
    """最后一条计算中取消：全部条目共 done，作业仍为 cancelled。"""

    client = app_worker.test_client()
    storage = app_worker.extensions["storage"]
    for round_ in range(3):
        ids = [_put(storage, f"last{round_}_{i}", n=40_000) for i in range(3)]
        jid = client.post("/api/jobs/spectrum", json={
            "record_ids": ids, "dampings": [0.05],
        }).get_json()["job_id"]

        _wait_item_status(client, jid, 2, ("running",))
        rv = client.post(f"/api/jobs/{jid}/cancel")
        assert rv.status_code == 200

        _wait_item_status(client, jid, 2, ("done",))
        _assert_status_stable(client, jid, "cancelled")

        job, items, counts = _consistent_via_api(client, jid)
        assert job["status"] == "cancelled"
        assert [i["status"] for i in items] == ["done"] * 3
        assert counts["done"] == 3
        assert job["result"] is None, "取消优先：汇总结果不落库"


def test_cancel_match_job_during_spectrum_phase(app_worker):
    """匹配作业的逐条算谱阶段取消，遵守同一套规则。"""

    client = app_worker.test_client()
    storage = app_worker.extensions["storage"]
    for round_ in range(2):
        ids = [_put(storage, f"mspec{round_}_{i}", n=40_000) for i in range(3)]
        jid = client.post("/api/jobs/match",
                          json=_match_payload(ids)).get_json()["job_id"]

        _wait_item_status(client, jid, 0, ("running",))
        rv = client.post(f"/api/jobs/{jid}/cancel")
        assert rv.status_code == 200

        _wait_item_status(client, jid, 0, ("done",))
        _assert_status_stable(client, jid, "cancelled")

        job, items, counts = _consistent_via_api(client, jid)
        assert [i["status"] for i in items] == ["done", "cancelled", "cancelled"]
        assert counts["done"] == 1 and counts["cancelled"] == 2


def test_cancel_during_matching_phase(app_worker, monkeypatch):
    """取消落在最后的排序匹配阶段：匹配结果丢弃，作业为 cancelled。

    用 monkeypatch 把 match_batch 换成「进入即阻塞」的版本，确定性地
    把取消送进匹配阶段窗口，反复打 3 次。
    """

    client = app_worker.test_client()
    storage = app_worker.extensions["storage"]
    ids = [_put(storage, f"match{i}", n=20_000) for i in range(3)]

    real_match_batch = jobs_mod.match_batch
    gate = {"entered": None, "release": None}

    def blocking_match_batch(*args, **kwargs):
        gate["entered"].set()
        assert gate["release"].wait(timeout=10), "测试未及时放行 match_batch"
        return real_match_batch(*args, **kwargs)

    monkeypatch.setattr(jobs_mod, "match_batch", blocking_match_batch)

    for _ in range(3):
        gate["entered"] = threading.Event()
        gate["release"] = threading.Event()
        jid = client.post("/api/jobs/match",
                          json=_match_payload(ids)).get_json()["job_id"]

        assert gate["entered"].wait(timeout=30), "作业未进入匹配阶段"
        rv = client.post(f"/api/jobs/{jid}/cancel")
        assert rv.status_code == 200
        gate["release"].set()

        _assert_status_stable(client, jid, "cancelled")
        job, items, counts = _consistent_via_api(client, jid)
        assert [i["status"] for i in items] == ["done"] * 3
        assert counts["done"] == 3
        assert job["result"] is None, "取消落在匹配阶段：匹配结果不落库"


def test_cancel_at_result_write_boundary(tmp_path, monkeypatch):
    """最后一条已算完、汇总结果尚未落库的瞬间来取消：判给取消。

    把 complete_job 换成「进入即阻塞」的版本，在结果落库前一刻取消，
    验证守卫式 UPDATE 让取消原子胜出。
    """

    storage = Storage(str(tmp_path / "boundary.db"))
    _put(storage, "r1", n=40_000)
    storage.create_job("j1", "spectrum", _spectrum_params(), total=1)
    storage.add_job_items("j1", [("r1", 0)])

    entered = threading.Event()
    release = threading.Event()
    orig_complete = storage.complete_job

    def blocking_complete(job_id, result):
        entered.set()
        release.wait(timeout=10)
        return orig_complete(job_id, result)

    monkeypatch.setattr(storage, "complete_job", blocking_complete)
    sch = Scheduler(storage)
    sch.start()
    try:
        assert entered.wait(timeout=30), "作业未走到结果落库"
        assert storage.request_cancel("j1") is True
        release.set()
        time.sleep(0.5)  # 等落库尝试与收尾结束
        job, items, counts = _consistent_via_storage(storage, "j1")
        assert job["status"] == "cancelled"
        assert job["result"] is None
        assert [i["status"] for i in items] == ["done"]
    finally:
        release.set()
        sch.stop()
        storage.close()


# ------------------------------------------------------- 取消的应答语义

def test_cancel_and_complete_are_atomic(storage):
    """取消与完成必有一方原子胜出，不存在中间态。"""

    _put(storage, "r1", n=200)

    # 取消先落库：完成写入被守卫拒绝
    storage.create_job("j1", "spectrum", _spectrum_params(), total=1)
    storage.add_job_items("j1", [("r1", 0)])
    storage.claim_next_queued()
    assert storage.request_cancel("j1") is True
    assert storage.complete_job("j1", {"x": 1}) is False
    job = storage.get_job("j1")
    assert job["status"] == "cancelled" and job["result"] is None

    # 完成先落库：取消返回 False（接口层据此回 409）
    storage.create_job("j2", "spectrum", _spectrum_params(), total=1)
    storage.add_job_items("j2", [("r1", 0)])
    storage.claim_next_queued()
    assert storage.complete_job("j2", {"x": 1}) is True
    assert storage.request_cancel("j2") is False
    assert storage.get_job("j2")["status"] == "completed"

    # 失败落库同样不得覆盖取消
    storage.create_job("j3", "spectrum", _spectrum_params(), total=1)
    storage.add_job_items("j3", [("r1", 0)])
    storage.claim_next_queued()
    assert storage.request_cancel("j3") is True
    assert storage.fail_job("j3", "boom") is False
    assert storage.get_job("j3")["status"] == "cancelled"


def test_item_running_guard_blocks_after_cancel(storage):
    """取消先把条目收尾后，执行线程不能再把它翻回 running。"""

    _put(storage, "r1", n=200)
    storage.create_job("j1", "spectrum", _spectrum_params(), total=1)
    storage.add_job_items("j1", [("r1", 0)])
    storage.claim_next_queued()
    assert storage.request_cancel("j1") is True
    assert storage.set_item_running("j1", 0) is False
    assert storage.list_job_items("j1")[0]["status"] == "cancelled"


def test_cancel_queued_job(app_worker):
    """排队中的作业取消后直接结束，条目全部为 cancelled，不占 worker。"""

    client = app_worker.test_client()
    storage = app_worker.extensions["storage"]
    rid_big = _put(storage, "occupy", n=150_000)   # 占住 worker 约 1.5s
    rid_small = _put(storage, "queued", n=2_000)

    j1 = client.post("/api/jobs/spectrum", json={
        "record_ids": [rid_big], "dampings": [0.05],
    }).get_json()["job_id"]
    _wait_job_status(client, j1, ("running",))

    j2 = client.post("/api/jobs/spectrum", json={
        "record_ids": [rid_small], "dampings": [0.05],
    }).get_json()["job_id"]
    rv = client.post(f"/api/jobs/{j2}/cancel")
    assert rv.status_code == 200

    job2, items2, counts2 = _consistent_via_api(client, j2)
    assert job2["status"] == "cancelled"
    assert [i["status"] for i in items2] == ["cancelled"]
    assert counts2["cancelled"] == 1

    # 前面的作业不受影响，正常完成
    wait_for_status(client, j1)
    job1, _, _ = _consistent_via_api(client, j1)
    assert job1["status"] == "completed"


def test_cancel_completed_job_returns_409(app_worker):
    """已完成的作业取消：如实回 409，结果不受影响。"""

    client = app_worker.test_client()
    storage = app_worker.extensions["storage"]
    rid = _put(storage, "done", n=2_000)
    jid = client.post("/api/jobs/spectrum", json={
        "record_ids": [rid], "dampings": [0.05],
    }).get_json()["job_id"]
    wait_for_status(client, jid)

    rv = client.post(f"/api/jobs/{jid}/cancel")
    assert rv.status_code == 409
    job, _, _ = _consistent_via_api(client, jid)
    assert job["status"] == "completed" and job["result"] is not None


def test_double_cancel_second_returns_409(app_worker):
    """重复取消：第一次 200，第二次 409（已处终态）。"""

    client = app_worker.test_client()
    storage = app_worker.extensions["storage"]
    rid = _put(storage, "twice", n=60_000)
    jid = client.post("/api/jobs/spectrum", json={
        "record_ids": [rid], "dampings": [0.05],
    }).get_json()["job_id"]
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 200
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 409
    _assert_status_stable(client, jid, "cancelled")
    _consistent_via_api(client, jid)


def test_resubmit_after_cancel_gives_identical_results(app_worker):
    """取消后按相同参数重新提交：结果与另一次全新提交完全一致。"""

    client = app_worker.test_client()
    storage = app_worker.extensions["storage"]
    ids = [_put(storage, f"redo{i}", n=20_000) for i in range(2)]
    payload = {"record_ids": ids, "dampings": [0.02, 0.05]}

    j_cancel = client.post("/api/jobs/spectrum", json=payload).get_json()["job_id"]
    rv = client.post(f"/api/jobs/{j_cancel}/cancel")
    assert rv.status_code == 200

    results = []
    for _ in range(2):
        jid = client.post("/api/jobs/spectrum", json=payload).get_json()["job_id"]
        body = wait_for_status(client, jid)
        assert body["job"]["status"] == "completed"
        results.append(body["job"]["result"])
    assert json.dumps(results[0], sort_keys=True) == json.dumps(
        results[1], sort_keys=True)

    # 被取消的作业保持 cancelled，不与新作业混淆
    _assert_status_stable(client, j_cancel, "cancelled")
    _consistent_via_api(client, j_cancel)


# ------------------------------------------------------- 失败与中断

def test_failed_job_items_aborted(tmp_path):
    """作业级异常失败：未跑条目收尾为 aborted，终态不变式成立。"""

    storage = Storage(str(tmp_path / "fail.db"))
    _put(storage, "f0", n=2_000)
    _put(storage, "f1", n=2_000)
    storage.create_job("jf", "spectrum", _spectrum_params(), total=2)
    storage.add_job_items("jf", [("f0", 0), ("f1", 1)])
    # 把参数改成合法 JSON 但非法取值（周期非递增），让执行线程在准备
    # 参数时抛出（模拟作业级失败）；get_job 本身不受影响
    storage.conn.execute(
        "UPDATE jobs SET params=? WHERE id='jf'",
        (json.dumps({**_spectrum_params(), "periods": [0.5, 0.2]}),),
    )
    storage.conn.commit()

    sch = Scheduler(storage)
    sch.start()
    try:
        job = _wait_storage_job(storage, "jf", ("failed",))
        assert "作业执行失败" in job["error"]
        _, items, counts = _consistent_via_storage(storage, "jf")
        assert [i["status"] for i in items] == ["aborted", "aborted"]
        assert counts["aborted"] == 2
    finally:
        sch.stop()
        storage.close()


def test_recovery_finalizes_interrupted_items(tmp_path):
    """执行中进程死亡：重启后未跑完条目为 interrupted，已算完结果可读。"""

    path = str(tmp_path / "crash.db")
    s1 = Storage(path)
    for i in range(3):
        _put(s1, f"c{i}", n=2_000)
    s1.create_job("jc", "spectrum", _spectrum_params(), total=3)
    s1.add_job_items("jc", [("c0", 0), ("c1", 1), ("c2", 2)])
    s1.claim_next_queued()
    # 模拟进程在第二条计算中死亡：第一条 done（含结果），第二条 running
    s1.set_item_running("jc", 0)
    s1.set_item_result("jc", 0, {"seq": 0, "record_id": "c0", "status": "done",
                                 "spectra": [{"damping": 0.05}]})
    s1.set_item_running("jc", 1)
    s1.close()

    s2 = Storage(path)
    sch = Scheduler(s2)
    sch.start()
    try:
        job, items, counts = _consistent_via_storage(s2, "jc")
        assert job["status"] == "interrupted"
        assert [i["status"] for i in items] == ["done", "interrupted", "interrupted"]
        assert items[0]["result"]["spectra"][0]["damping"] == 0.05, (
            "中断前已算完的条目结果在重启后必须仍可读"
        )
        assert counts["done"] == 1 and counts["interrupted"] == 2
    finally:
        sch.stop()
        s2.close()


def test_recovery_finalizes_cancelled_job_leftovers(tmp_path):
    """崩溃发生在取消落库后、执行线程收尾前：重启把遗留条目补为 cancelled。"""

    path = str(tmp_path / "crash_cancel.db")
    s1 = Storage(path)
    for i in range(3):
        _put(s1, f"x{i}", n=2_000)
    s1.create_job("jx", "spectrum", _spectrum_params(), total=3)
    s1.add_job_items("jx", [("x0", 0), ("x1", 1), ("x2", 2)])
    s1.claim_next_queued()
    s1.set_item_running("jx", 0)
    s1.set_item_result("jx", 0, {"seq": 0, "record_id": "x0", "status": "done"})
    s1.set_item_running("jx", 1)
    # 取消落库（pending 的 x2 被收尾为 cancelled，running 的 x1 保留），
    # 随后进程在执行线程收尾前死亡
    assert s1.request_cancel("jx") is True
    s1.close()

    s2 = Storage(path)
    sch = Scheduler(s2)
    sch.start()
    try:
        job, items, counts = _consistent_via_storage(s2, "jx")
        assert job["status"] == "cancelled"
        assert [i["status"] for i in items] == ["done", "cancelled", "cancelled"]
        assert counts["done"] == 1 and counts["cancelled"] == 2
    finally:
        sch.stop()
        s2.close()


# ------------------------------------------------------- 真·杀进程重启

_CHILD_SCRIPT = """
import sys, time
import numpy as np
from app.scheduler import Scheduler
from app.storage import Storage

path = sys.argv[1]
s = Storage(path)
for i in range(3):
    t = np.arange(150_000) * 0.005
    acc = 0.3 * np.sin(2 * np.pi * 1.5 * t)
    s.upsert_record(f"r{i}", name=f"r{i}.txt", dt=0.005, unit_input="g",
                    fmt="single_column", metadata={}, acc=acc, status="ready")
s.create_job("jkill", "spectrum", {
    "periods": np.logspace(np.log10(0.02), np.log10(6), 100).tolist(),
    "dampings": [0.05], "method": "average_acceleration",
    "instability_policy": "refine", "baseline_mode": "mean"}, total=3)
s.add_job_items("jkill", [("r0", 0), ("r1", 1), ("r2", 2)])
sch = Scheduler(s)
sch.start()
print("READY", flush=True)
time.sleep(120)
"""


def test_hard_kill_mid_execution_then_recover(tmp_path):
    """子进程执行作业时被 SIGKILL（模拟容器执行中重启）。

    重启后：作业 interrupted；条目无 pending/running；第一条（已算完）
    结果仍可读；其余条目为 interrupted；计数三者一致。
    """

    db_path = str(tmp_path / "kill.db")
    script = tmp_path / "worker_child.py"
    script.write_text(_CHILD_SCRIPT)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.Popen(
        [sys.executable, str(script), db_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, cwd=str(REPO_ROOT),
    )
    storage = None
    try:
        line = proc.stdout.readline()
        assert line.strip() == "READY", f"子进程未就绪：{line!r}"

        storage = Storage(db_path)
        # 等第一条算完落库，此时第二条正在计算
        deadline = time.time() + 60
        while True:
            items = storage.list_job_items("jkill")
            if items and items[0]["status"] == "done":
                break
            assert time.time() < deadline, "子进程迟迟未算完第一条"
            assert proc.poll() is None, "子进程提前退出"
            time.sleep(0.05)

        proc.kill()  # SIGKILL，等价于容器执行中被杀掉
        proc.wait(timeout=10)
        assert proc.returncode == -signal.SIGKILL

        # 「重启」：新调度器启动，做崩溃恢复
        sch = Scheduler(storage)
        sch.start()
        try:
            job, items, counts = _consistent_via_storage(storage, "jkill")
            assert job["status"] == "interrupted"
            assert counts["pending"] == 0 and counts["running"] == 0
            assert items[0]["status"] == "done"
            assert items[0]["result"] is not None, (
                "中断前已算完的条目结果在重启后必须仍可读"
            )
            assert items[0]["result"]["spectra"], "条目结果应包含谱数据"
            assert [i["status"] for i in items[1:]] == ["interrupted"] * 2
            assert counts["done"] == 1 and counts["interrupted"] == 2
        finally:
            sch.stop()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        if storage is not None:
            storage.close()

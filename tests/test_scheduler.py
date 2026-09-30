"""调度器测试：后台执行、取消轮询、崩溃恢复后继续排队。"""

from __future__ import annotations

import time

import numpy as np

from app.scheduler import Scheduler
from app.storage import Storage


def _put(storage, rid, n=200, dt=0.01):
    storage.upsert_record(
        rid, name=f"{rid}.txt", dt=dt, unit_input="m/s2", fmt="single_column",
        metadata={}, acc=np.sin(np.linspace(0, 10, n)), status="ready",
    )


def _wait(job_getter, jid, targets=("completed", "failed", "cancelled"),
          timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = job_getter(jid)
        if job["status"] in targets:
            return job
        time.sleep(0.02)
    raise AssertionError(f"{jid} 未在 {timeout}s 内进入 {targets}")


def test_spectrum_job_runs_in_background(tmp_path):
    storage = Storage(str(tmp_path / "s.db"))
    _put(storage, "r1")
    storage.create_job("j1", "spectrum", {
        "periods": [0.1, 0.5, 1.0, 2.0], "dampings": [0.05],
        "method": "average_acceleration", "instability_policy": "refine",
        "baseline_mode": "mean",
    }, total=1)
    storage.add_job_items("j1", [("r1", 0)])

    sch = Scheduler(storage)
    sch.start()
    try:
        job = _wait(storage.get_job, "j1")
        assert job["status"] == "completed"
        assert job["progress"] == 1
        items = storage.list_job_items("j1")
        assert items[0]["status"] == "done"
        assert "spectra" in items[0]["result"]
    finally:
        sch.stop()
        storage.close()


def test_cancel_running_job(tmp_path):
    storage = Storage(str(tmp_path / "c.db"))
    for i in range(3):
        _put(storage, f"r{i}", n=60000)  # 较大记录，每条算一会儿
    storage.create_job("big", "spectrum", {
        "periods": np.linspace(0.05, 6, 100).tolist(),
        "dampings": [0.05],
        "method": "average_acceleration", "instability_policy": "refine",
        "baseline_mode": "mean",
    }, total=3)
    storage.add_job_items("big", [(f"r{i}", i) for i in range(3)])

    sch = Scheduler(storage)
    sch.start()
    try:
        time.sleep(0.15)
        assert storage.request_cancel("big") is True
        sch.notify()
        job = _wait(storage.get_job, "big")
        assert job["status"] == "cancelled"
    finally:
        sch.stop()
        storage.close()


def test_recovery_queued_continues_after_restart(tmp_path):
    """j1 中断、j2 排队：新调度器启动后 j2 正常执行完成。"""

    path = str(tmp_path / "r.db")
    s1 = Storage(path)
    _put(s1, "r1")
    s1.create_job("j1", "spectrum", {"periods": [0.5], "dampings": [0.05],
                                     "method": "average_acceleration",
                                     "instability_policy": "refine",
                                     "baseline_mode": "mean"}, total=1)
    s1.add_job_items("j1", [("r1", 0)])
    s1.claim_next_queued()  # 模拟 j1 正在执行时进程死亡
    s1.create_job("j2", "spectrum", {"periods": [0.5], "dampings": [0.05],
                                     "method": "average_acceleration",
                                     "instability_policy": "refine",
                                     "baseline_mode": "mean"}, total=1)
    s1.add_job_items("j2", [("r1", 0)])
    s1.close()

    s2 = Storage(path)
    sch = Scheduler(s2)
    sch.start()
    try:
        j1 = _wait(s2.get_job, "j1", targets=("interrupted",))
        assert j1["status"] == "interrupted"
        j2 = _wait(s2.get_job, "j2")
        assert j2["status"] == "completed"
    finally:
        sch.stop()
        s2.close()


def test_single_bad_record_does_not_fail_job(tmp_path):
    storage = Storage(str(tmp_path / "b.db"))
    _put(storage, "good")
    # good 之后再放一条被标记 error 的记录到作业项（模拟上传即失败的引用）
    storage.upsert_record(
        "broken", name="broken.txt", dt=1.0, unit_input=None, fmt="unknown",
        metadata={}, acc=None, status="error", error="非数值行",
    )
    storage.create_job("jmix", "spectrum", {"periods": [0.5], "dampings": [0.05],
                                            "method": "average_acceleration",
                                            "instability_policy": "refine",
                                            "baseline_mode": "mean"}, total=2)
    storage.add_job_items("jmix", [("good", 0), ("broken", 1)])
    sch = Scheduler(storage)
    sch.start()
    try:
        job = _wait(storage.get_job, "jmix")
        assert job["status"] == "completed"
        result = job["result"]
        assert result["summary"]["done"] == 1
        assert result["summary"]["skipped"] == 1
    finally:
        sch.stop()
        storage.close()

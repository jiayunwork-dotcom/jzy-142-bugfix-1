"""SQLite 存储层测试：落库读取、崩溃恢复、作业项进度。"""

from __future__ import annotations

import numpy as np

from app.storage import Storage


def _put_ready(storage: Storage, rid="rec1", n=100, dt=0.01, unit="m/s2"):
    storage.upsert_record(
        rid, name=f"{rid}.txt", dt=dt, unit_input=unit, fmt="single_column",
        metadata={"station": "S1"}, acc=np.linspace(0, 1, n), status="ready",
    )


def test_record_roundtrip(storage):
    _put_ready(storage, "recX", n=500, dt=0.005, unit="g")
    rec = storage.get_record("recX")
    assert rec is not None
    assert rec.npts == 500
    assert rec.duration == 0.005 * 499
    assert rec.unit_input == "g"
    assert rec.metadata["station"] == "S1"
    np.testing.assert_array_equal(rec.acc, np.linspace(0, 1, 500))


def test_error_record_persisted(storage):
    storage.upsert_record(
        "rec_bad1", name="bad.txt", dt=1.0, unit_input=None, fmt="unknown",
        metadata={}, acc=None, status="error", error="空文件",
    )
    rec = storage.get_record("rec_bad1")
    assert rec.status == "error"
    assert rec.error == "空文件"
    assert rec.acc.size == 0
    listed = storage.list_records()
    assert listed[0]["status"] == "error"


def test_job_lifecycle_and_items(storage):
    _put_ready(storage, "a")
    _put_ready(storage, "b")
    storage.create_job("job1", "spectrum", {"dampings": [0.05]}, total=2)
    storage.add_job_items("job1", [("a", 0), ("b", 1)])

    assert storage.claim_next_queued() == "job1"
    assert storage.claim_next_queued() is None
    job = storage.get_job("job1")
    assert job["status"] == "running"

    storage.set_item_running("job1", 0)
    storage.set_item_result("job1", 0, {"sd": [1, 2]})
    storage.set_job_progress("job1", 1)
    storage.set_item_error("job1", 1, "积分失败")
    storage.set_job_progress("job1", 2)

    items = storage.list_job_items("job1")
    assert items[0]["status"] == "done"
    assert items[0]["result"] == {"sd": [1, 2]}
    assert items[1]["status"] == "error"
    assert items[1]["error"] == "积分失败"
    counts = storage.counts_by_status("job1")
    assert counts["done"] == 1 and counts["error"] == 1

    storage.complete_job("job1", {"summary": {"ok": True}})
    assert storage.get_job("job1")["status"] == "completed"
    assert storage.get_job("job1")["result"]["summary"] == {"ok": True}


def test_cancel_queued_and_running(storage):
    _put_ready(storage, "a")
    storage.create_job("j", "spectrum", {}, total=1)
    storage.add_job_items("j", [("a", 0)])
    # queued 直接取消
    assert storage.request_cancel("j") is True
    assert storage.get_job("j")["status"] == "cancelled"
    assert storage.request_cancel("j") is False


def test_recover_interrupted(tmp_path):
    path = str(tmp_path / "rec.db")
    s1 = Storage(path)
    _put_ready(s1, "a")
    s1.create_job("j1", "spectrum", {}, total=1)
    s1.add_job_items("j1", [("a", 0)])
    assert s1.claim_next_queued() == "j1"  # running
    s1.create_job("j2", "spectrum", {}, total=1)
    s1.add_job_items("j2", [("a", 0)])   # queued
    s1.close()

    # 模拟进程重启
    s2 = Storage(path)
    interrupted = s2.recover_interrupted()
    assert interrupted == ["j1"]
    assert s2.get_job("j1")["status"] == "interrupted"
    # queued 作业仍可被领取
    assert s2.claim_next_queued() == "j2"
    s2.close()


def test_results_survive_restart(tmp_path):
    path = str(tmp_path / "persist.db")
    s1 = Storage(path)
    _put_ready(s1, "a")
    s1.create_job("jj", "spectrum", {}, total=1)
    s1.add_job_items("jj", [("a", 0)])
    assert s1.claim_next_queued() == "jj"
    s1.complete_job("jj", {"answer": 42, "nested": {"x": [1, 2, 3]}})
    s1.close()

    s2 = Storage(path)
    job = s2.get_job("jj")
    assert job["status"] == "completed"
    assert job["result"]["answer"] == 42
    assert job["result"]["nested"]["x"] == [1, 2, 3]
    assert s2.get_record("a").npts == 100
    s2.close()

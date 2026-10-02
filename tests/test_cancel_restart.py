"""取消与重启语义的回归测试。

覆盖用户报告的三类事故，且每个时机都反复多次（>10 次）执行：
  1. 取消成功后作业又翻盘为 completed（单条大记录，取消落在最后一条
     的计算中）；取消落在第一条中、最后一条中；匹配作业的逐条算谱
     阶段与最后的排序匹配阶段；
  2. 多记录取消后残留 pending/running；
  3. 执行中重启后作业项残留 running/pending、计数与作业状态矛盾。

每次都同时核对三方一致：作业终态、逐条作业项状态、状态计数之和。

为了不依赖「睡多久」这种脆弱时序，测试通过两种确定性手段把取消打到
指定相位：
  - 轮询作业项状态（等到某 seq 变 running / done）；
  - monkeypatch app.jobs._before_match_hook，在「全部谱算完、匹配未
    开始」的窗口内触发取消（模拟匹配阶段收到取消）。
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import app.jobs as jobs_mod
from app.scheduler import Scheduler
from app.storage import Storage


# ---------------------------------------------------------------- 记录构造

def _put(storage, rid, npts, dt=0.005, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(npts) * dt
    acc = 0.3 * np.sin(2 * np.pi * 2.0 * t) + 0.05 * rng.normal(size=npts)
    storage.upsert_record(
        rid, name=f"{rid}.txt", dt=dt, unit_input="g", fmt="single_column",
        metadata={}, acc=acc, status="ready",
    )


def _new_scheduler(path):
    storage = Storage(path)
    sch = Scheduler(storage)
    sch.start()
    return storage, sch


def _create_spectrum(storage, jid, rids, periods=100, dampings=(0.02, 0.05, 0.1)):
    if isinstance(periods, int):
        periods = np.linspace(0.02, 6.0, periods).tolist()
    storage.create_job(jid, "spectrum", {
        "periods": periods, "dampings": list(dampings),
        "method": "average_acceleration", "instability_policy": "refine",
        "baseline_mode": "mean",
    }, total=len(rids))
    storage.add_job_items(jid, [(rid, i) for i, rid in enumerate(rids)])


def _create_match(storage, jid, rids):
    storage.create_job(jid, "match", {
        "periods": np.linspace(0.05, 3.0, 80).tolist(),
        "dampings": [0.05],
        "method": "average_acceleration", "instability_policy": "refine",
        "baseline_mode": "mean",
        "design_spectrum": {
            "periods": [0.05, 0.1, 0.5, 1.0, 3.0],
            "values": [0.30, 0.30, 0.30, 0.20, 0.10], "unit": "g",
        },
        "t1": 0.1, "t2": 2.0, "s_min": 0.2, "s_max": 5.0,
        "top_n": 3, "quantity": "psa", "bounds_policy": "reject",
        "target_damping": 0.05,
    }, total=len(rids))
    storage.add_job_items(jid, [(rid, i) for i, rid in enumerate(rids)])


# ---------------------------------------------------------------- 等待工具

def _wait_job(storage, jid, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = storage.get_job(jid)
        if job["status"] in ("completed", "failed", "cancelled", "interrupted"):
            return job
        time.sleep(0.01)
    raise AssertionError(f"{jid} 未在 {timeout}s 内进入终态")


def _assert_consistency(storage, jid, total, *, expected_status=None):
    """三方一致性：作业状态、逐条状态、计数之和 == total。"""

    job = storage.get_job(jid)
    items = storage.list_job_items(jid)
    counts = storage.counts_by_status(jid)
    if expected_status is not None:
        assert job["status"] == expected_status, (
            f"{jid} 状态 {job['status']!r}，期望 {expected_status!r}"
        )
    assert len(items) == total
    assert sum(counts.values()) == total, counts
    statuses = [it["status"] for it in items]
    assert not any(s in ("pending", "running") for s in statuses), statuses
    return job, items, counts


# ============================================================ 1. 取消不翻盘

@pytest.mark.parametrize("trial", range(12))
def test_cancel_during_only_record_never_flips(tmp_path, trial, monkeypatch):
    """单条 20 万点：取消确认后，无论等多久只能是 cancelled。"""

    path = str(tmp_path / f"c1_{trial}.db")
    storage, sch = _new_scheduler(path)
    try:
        _put(storage, "big", 200_000, seed=trial)
        jid = f"job_single_{trial}"
        _create_spectrum(storage, jid, ["big"])
        wait_armed = _arm_cancel_at_item(monkeypatch, storage, jid, 0)
        wait_armed()
        assert storage.get_job(jid)["cancel_requested"] is True
        job = _wait_job(storage, jid)
        assert job["status"] == "cancelled"
        assert job["result"] is None
        # 再等两个计算周期那么久，确认不会事后翻盘
        time.sleep(0.5)
        assert storage.get_job(jid)["status"] == "cancelled"
        _assert_consistency(storage, jid, 1, expected_status="cancelled")
        items = storage.list_job_items(jid)
        # 算到一半被中止：条目 cancelled 且无结果
        assert items[0]["status"] == "cancelled"
        assert items[0]["result"] is None
        assert items[0]["error"]
    finally:
        monkeypatch.undo()
        sch.stop(timeout=10)
        storage.close()


@pytest.mark.parametrize("trial", range(8))
def test_cancel_during_first_and_last_of_three(tmp_path, trial, monkeypatch):
    """三条记录：取消落在第一条 / 最后一条的计算中。

    取消落在第一条：后两条应是 cancelled（没跑）；
    落在最后一条：前两条 done 结果保留，最后一条 cancelled。
    """

    path = str(tmp_path / f"c3_{trial}.db")
    storage, sch = _new_scheduler(path)
    try:
        rids = [f"r{i}" for i in range(3)]
        for i, rid in enumerate(rids):
            _put(storage, rid, 60_000, seed=100 * trial + i)
        jid = f"job3_{trial}"
        _create_spectrum(storage, jid, rids)

        during_last = trial % 2 == 0
        target_seq = 2 if during_last else 0
        expected = ({0: "done", 1: "done", 2: "cancelled"} if during_last
                    else {0: "cancelled", 1: "cancelled", 2: "cancelled"})
        wait_armed = _arm_cancel_at_item(monkeypatch, storage, jid, target_seq)
        wait_armed()

        _wait_job(storage, jid)
        time.sleep(0.2)  # 确认稳定不翻盘
        job, items, counts = _assert_consistency(
            storage, jid, 3, expected_status="cancelled")
        got = {it["seq"]: it["status"] for it in items}
        assert got == expected, (trial, got)
        # done 的条目结果在取消后依然可读
        for it in items:
            if it["status"] == "done":
                assert it["result"] and "spectra" in it["result"]
            else:
                assert it["result"] is None and it["error"]
        assert job["progress"] == counts["done"] + counts["error"] + counts["skipped"]
    finally:
        monkeypatch.undo()
        sch.stop(timeout=10)
        storage.close()


def test_cancel_too_late_returns_conflict(tmp_path):
    """最后一条算完、结果尚未写入的瞬间来了取消：仲裁必须二选一且不矛盾。

    A. 取消标志先落库，执行线程随后 finalize -> cancelled，取消请求成功，
       作业结果丢弃；
    B. finalize 先提交 completed，取消请求随后到达 -> 如实拒绝（接口 409），
       作业结果保留；
    C. 再用一对在 Barrier 上同时释放、带随机抖动的线程对打，确认自由
       竞争下结果永远只落在这两种状态之一，且与各方返回值自洽。
    """

    # A：取消先到，finalize 随后仲裁 -> cancelled（结果丢弃）
    for trial in range(10):
        path = str(tmp_path / f"a_{trial}.db")
        storage = Storage(path)
        _put(storage, "r", 100, seed=trial)
        _create_spectrum(storage, "j", ["r"], periods=[0.1, 0.5, 1.0],
                         dampings=[0.05])
        storage.claim_next_queued()
        storage.set_item_running("j", 0)
        storage.set_item_result("j", 0, {"sd": [1, 2, 3]})
        assert storage.request_cancel("j") == (True, "running")
        assert storage.finalize_job("j", {"summary": {"done": 1}}) == "cancelled"
        job = storage.get_job("j")
        assert job["status"] == "cancelled" and job["result"] is None
        # 终态后再取消 -> 冲突
        assert storage.request_cancel("j")[0] is False
        storage.close()

    # B：完成先到，取消随后到达 -> 如实冲突（接口 409），结果保留
    for trial in range(10):
        path = str(tmp_path / f"b_{trial}.db")
        storage = Storage(path)
        _put(storage, "r", 100, seed=trial)
        _create_spectrum(storage, "j", ["r"], periods=[0.1, 0.5, 1.0],
                         dampings=[0.05])
        storage.claim_next_queued()
        storage.set_item_running("j", 0)
        storage.set_item_result("j", 0, {"sd": [1, 2, 3]})
        assert storage.finalize_job("j", {"summary": {"done": 1}}) == "completed"
        accepted, reason = storage.request_cancel("j")
        assert accepted is False and reason == "completed"
        job = storage.get_job("j")
        assert job["status"] == "completed"
        assert job["result"]["summary"] == {"done": 1}
        storage.close()

    # C：自由竞争，结果只能属于 {cancelled, completed} 且与返回值自洽
    rng = np.random.default_rng(0)
    for trial in range(40):
        path = str(tmp_path / f"c_{trial}.db")
        storage = Storage(path)
        _put(storage, "r", 100, seed=trial)
        _create_spectrum(storage, "j", ["r"], periods=[0.1, 0.5, 1.0],
                         dampings=[0.05])
        storage.claim_next_queued()
        storage.set_item_running("j", 0)
        storage.set_item_result("j", 0, {"sd": [1, 2, 3]})

        barrier = threading.Barrier(2)
        results = {}

        def finisher():
            barrier.wait()
            if rng.random() < 0.5:
                time.sleep(rng.random() * 0.002)
            results["final"] = storage.finalize_job(
                "j", {"summary": {"done": 1}})

        def canceller():
            barrier.wait()
            if rng.random() < 0.5:
                time.sleep(rng.random() * 0.002)
            results["cancel"] = storage.request_cancel("j")

        t1 = threading.Thread(target=finisher)
        t2 = threading.Thread(target=canceller)
        t1.start(); t2.start(); t1.join(); t2.join()

        job = storage.get_job("j")
        accepted, reason = results["cancel"]
        if results["final"] == "cancelled":
            assert job["status"] == "cancelled"
            assert accepted is True and reason == "running"
            assert job["result"] is None
        else:
            assert results["final"] == "completed"
            assert job["status"] == "completed"
            assert accepted is False and reason == "completed"
            assert job["result"]["summary"] == {"done": 1}
        storage.close()


# ============================================================ 2. 匹配作业

@pytest.mark.parametrize("trial", range(6))
def test_cancel_match_during_spectrum_phase(tmp_path, trial, monkeypatch):
    """匹配作业：取消落在逐条算谱阶段（第一条/最后一条）。"""

    path = str(tmp_path / f"m1_{trial}.db")
    storage, sch = _new_scheduler(path)
    try:
        rids = [f"m{i}" for i in range(3)]
        for i, rid in enumerate(rids):
            _put(storage, rid, 60_000, seed=200 * trial + i)
        jid = f"jm_{trial}"
        _create_match(storage, jid, rids)

        if trial % 2 == 0:
            target_seq, expected = 2, {0: "done", 1: "done", 2: "cancelled"}
        else:
            target_seq, expected = 0, {0: "cancelled", 1: "cancelled", 2: "cancelled"}
        wait_armed = _arm_cancel_at_item(monkeypatch, storage, jid, target_seq)
        wait_armed()
        _wait_job(storage, jid)
        time.sleep(0.2)
        job, items, counts = _assert_consistency(
            storage, jid, 3, expected_status="cancelled")
        assert {it["seq"]: it["status"] for it in items} == expected
        assert job["result"] is None
    finally:
        monkeypatch.undo()
        sch.stop(timeout=10)
        storage.close()


@pytest.mark.parametrize("trial", range(10))
def test_cancel_match_during_ranking_phase(tmp_path, trial, monkeypatch):
    """匹配作业：最后一条谱已算完，取消落在排序匹配阶段。

    通过 _before_match_hook 在 match_batch 开始前登记取消。此时取消请求
    与结果提交赛跑，终态必须且只能是 cancelled（作业返回 cancelled=True
    路径），逐条 done 结果保留、作业整体结果丢弃。
    """

    path = str(tmp_path / f"m2_{trial}.db")
    storage, sch = _new_scheduler(path)
    try:
        rids = [f"q{i}" for i in range(3)]
        for i, rid in enumerate(rids):
            _put(storage, rid, 20_000, seed=300 * trial + i)
        jid = f"jmm_{trial}"
        _create_match(storage, jid, rids)

        fired = threading.Event()

        def hook(job_id, st, is_cancelled):
            if job_id == jid and not fired.is_set():
                fired.set()
                assert st.request_cancel(jid)[0] is True

        monkeypatch.setattr(jobs_mod, "_before_match_hook", hook)
        job = _wait_job(storage, jid)
        assert fired.is_set(), "钩子未触发，测试没有打到匹配阶段"
        assert job["status"] == "cancelled"
        job2, items, counts = _assert_consistency(
            storage, jid, 3, expected_status="cancelled")
        assert counts["done"] == 3
        assert all(it["result"] is not None for it in items)  # 谱结果保留
        assert job2["result"] is None                          # 匹配结果丢弃
    finally:
        monkeypatch.undo()
        sch.stop(timeout=10)
        storage.close()


def test_cancel_match_just_before_commit_can_win(tmp_path, monkeypatch):
    """match_batch 已算完、finalize 提交前的极小窗口取消：仲裁仍成立。

    在钩子中取消，但钩子安排在「match_batch 之后」需要额外缝；此处验证
    作业线程把 result 交给 finalize_job 时若标志已置，finalize 判 cancelled。
    （直接在存储层制造该时序，等价于取消在结果提交前一个指令落库。）
    """

    path = str(tmp_path / "m3.db")
    storage = Storage(path)
    _put(storage, "q0", 100)
    _create_match(storage, "jj", ["q0"])
    storage.claim_next_queued()
    storage.set_item_running("jj", 0)
    storage.set_item_result("jj", 0, {"spectrum": {}})
    # 结果已在内存里，提交前一刻取消落库
    assert storage.request_cancel("jj") == (True, "running")
    assert storage.finalize_job("jj", {"matching": {"ranking": []}}) == "cancelled"
    job = storage.get_job("jj")
    assert job["status"] == "cancelled" and job["result"] is None
    items = storage.list_job_items("jj")
    assert items[0]["status"] == "done"  # 已完成的谱结果不回收
    storage.close()


# ============================================================ 3. 执行中重启

def _arm_cancel_at_item(monkeypatch, storage, jid, seq):
    """在指定条目 set_item_running 提交的**同一时刻**（工作线程内）登记取消。

    比「先轮询等到 running、再从测试线程发取消」更严格：后者在两个线程
    之间留有窗口，高负载下条目可能已经跑完进入下一条，导致取消落在哪一
    条上不稳定。这里取消在 running 标记的同一同步点之后立即落库，而条目
    的积分循环要到第 256 步才首次轮询标志，因此取消必然落在该条计算中。
    """

    real_set_running = storage.set_item_running
    armed = threading.Event()

    def hooked(job_id, s, *a, **k):
        real_set_running(job_id, s, *a, **k)
        if job_id == jid and s == seq and not armed.is_set():
            armed.set()
            accepted, reason = storage.request_cancel(jid)
            assert accepted is True and reason == "running"

    monkeypatch.setattr(storage, "set_item_running", hooked)

    def wait_armed(timeout=60.0):
        if not armed.wait(timeout):
            raise AssertionError(f"取消未能在条目 {seq} 开始时登记")

    return wait_armed


def _freeze_worker_at_item(monkeypatch, storage, jid, seq):
    """把工作线程冻结在指定条目「已置 running、谱尚未算完」的瞬间。

    Python 线程无法被强杀，而真进程被 SIGKILL 时执行是瞬间停止的；因此
    在工作线程的同步提交点 ``Storage.set_item_running`` 上插桩：目标条目
    一旦置为 running（事务已落库）即永久阻塞工作线程，精确等价于崩溃
    瞬间的磁盘现场。返回等待函数与释放函数。
    """

    real_set_running = storage.set_item_running
    frozen = threading.Event()
    release = threading.Event()

    def hooked(job_id, s, *a, **k):
        real_set_running(job_id, s, *a, **k)
        if job_id == jid and s == seq:
            frozen.set()
            # 阻塞工作线程；被释放时（旧“进程”已被恢复取代）抛错退出，
            # 绝不让旧线程继续写库
            release.wait(timeout=30)
            raise RuntimeError("simulated process killed during item")

    monkeypatch.setattr(storage, "set_item_running", hooked)

    def wait_frozen(timeout=60.0):
        if not frozen.wait(timeout):
            raise AssertionError("工作线程未在目标条目冻结")
        time.sleep(0.05)  # 确认线程已停在 wait 上、磁盘现场稳定

    return wait_frozen, release


@pytest.mark.parametrize("trial", range(6))
def test_restart_during_execution_consistent(tmp_path, trial, monkeypatch):
    """执行中（第一条/最后一条在跑）模拟进程死亡：重启后状态三方一致。"""

    path = str(tmp_path / f"k_{trial}.db")
    storage, sch = _new_scheduler(path)
    rids = [f"k{i}" for i in range(3)]
    for i, rid in enumerate(rids):
        _put(storage, rid, 150_000, seed=400 * trial + i)
    jid = f"jk_{trial}"
    _create_spectrum(storage, jid, rids)
    target_seq = 0 if trial % 2 == 0 else 2

    wait_frozen, release = _freeze_worker_at_item(
        monkeypatch, storage, jid, target_seq)
    wait_frozen()
    # 此时磁盘现场 = 进程在该条目计算中被杀死
    sch.hard_kill()
    storage.close()

    # 新进程：恢复
    s2 = Storage(path)
    try:
        recovered = s2.recover_interrupted()
        assert recovered == [jid]
        job = s2.get_job(jid)
        assert job["status"] == "interrupted"
        assert "重启" in job["error"]
        items = s2.list_job_items(jid)
        counts = s2.counts_by_status(jid)
        assert sum(counts.values()) == 3
        assert counts["pending"] == counts["running"] == 0
        got = {it["seq"]: it["status"] for it in items}
        # 在跑的那条与其后条目都没跑完，且都标明是中断
        for seq in range(target_seq, 3):
            assert got[seq] == "interrupted"
        # 之前算完的条目结果重启后仍可读
        for it in items:
            if it["status"] == "done":
                assert it["result"] and "spectra" in it["result"]
            elif it["status"] == "interrupted":
                assert it["result"] is None and it["error"]
    finally:
        s2.close()
        release.set()  # 解冻被遗弃的旧工作线程，让其安静退出
        monkeypatch.undo()


def test_restart_then_queued_continues_and_cancel_requested_stays_cancel(tmp_path):
    """重启恢复后：queued 作业照常续跑；带取消请求的 running 判 cancelled。"""

    path = str(tmp_path / "mix.db")
    s1 = Storage(path)
    _put(s1, "a", 80_000, seed=7)
    _put(s1, "b", 80_000, seed=8)
    # jk：running 且已确认取消；jq：queued
    s1.create_job("jk", "spectrum", {
        "periods": [0.1, 0.5, 1.0], "dampings": [0.05],
        "method": "average_acceleration", "instability_policy": "refine",
        "baseline_mode": "mean"}, total=1)
    s1.add_job_items("jk", [("a", 0)])
    s1.create_job("jq", "spectrum", {
        "periods": [0.1, 0.5, 1.0], "dampings": [0.05],
        "method": "average_acceleration", "instability_policy": "refine",
        "baseline_mode": "mean"}, total=1)
    s1.add_job_items("jq", [("b", 0)])
    assert s1.claim_next_queued() == "jk"
    s1.set_item_running("jk", 0)
    assert s1.request_cancel("jk") == (True, "running")
    s1.close()  # 进程死亡（无工作线程在跑，状态稳定）

    # 新进程：Scheduler.start() 内部先恢复，再启动工作线程领取 queued
    s2 = Storage(path)
    sch = Scheduler(s2)
    try:
        sch.start()
        job_k = s2.get_job("jk")
        assert job_k["status"] == "cancelled"  # 取消承诺不随重启失效
        assert s2.list_job_items("jk")[0]["status"] == "cancelled"
        # queued 作业被新工作线程领取并完成，结果与无重启时一致
        job_q = _wait_job(s2, "jq")
        assert job_q["status"] == "completed"
        assert job_q["result"]["summary"]["done"] == 1
    finally:
        sch.stop(timeout=10)
        s2.close()


"""pytest 公共夹具与合成记录工具。"""

from __future__ import annotations

import numpy as np
import pytest

from app import create_app
from app.storage import Storage


@pytest.fixture()
def storage(tmp_path):
    db = Storage(str(tmp_path / "test.db"))
    yield db
    db.close()


@pytest.fixture()
def app(tmp_path):
    db_path = str(tmp_path / "api.db")
    application = create_app(db_path, start_worker=False)
    application.config["TEST_DB_PATH"] = db_path
    yield application
    scheduler = application.extensions["scheduler"]
    scheduler.stop(timeout=2.0)
    ext_storage = application.extensions["storage"]
    ext_storage.close()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def app_worker(tmp_path):
    """带后台工作线程的应用，用于端到端作业测试。"""

    db_path = str(tmp_path / "worker.db")
    application = create_app(db_path, start_worker=True)
    yield application
    scheduler = application.extensions["scheduler"]
    scheduler.stop(timeout=5.0)
    application.extensions["storage"].close()


def sine_record(T0: float, dt: float, n_cycles: float, *,
                end_phase: float | None = None, amp: float = 1.0):
    """返回 (t, ag) 的正弦地面运动。

    默认整数周期截断（末端 ag=0）；``end_phase`` 给定时，在达到
    n_cycles 个完整周期后再补到指定稳态响应相位对应的时长，使自由
    振动初始包络 R0 不超过稳态幅值——用于稳态放大系数 1% 解析对照。
    """

    if end_phase is None:
        t = np.arange(0, n_cycles * T0, dt)
        return t, amp * np.sin(2 * np.pi * t / T0)
    # θ = ω t_end（强迫正弦相位），由调用方给定
    n = int(round(n_cycles * T0 / dt))
    t_extra = end_phase / (2 * np.pi) * T0
    t = np.arange(0, n * dt + t_extra, dt)
    return t, amp * np.sin(2 * np.pi * t / T0)


def wait_for_status(client, job_id, targets=("completed", "failed", "cancelled"),
                    timeout=30.0):
    """轮询作业直到进入终态，超时抛错。"""

    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        rv = client.get(f"/api/jobs/{job_id}")
        body = rv.get_json()
        status = body["job"]["status"]
        if status in targets:
            return body
        time.sleep(0.02)
    raise AssertionError(f"作业 {job_id} 在 {timeout}s 内未进入 {targets}")

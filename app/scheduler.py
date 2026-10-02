"""后台作业调度：单工作线程串行执行排队作业。

设计上刻意保持简单：一个 daemon 线程、一次只跑一个作业（反应谱计算
本身已在 NumPy 层并行，且 SQLite 单写），作业内部在每条记录处理前
轮询取消状态。取消请求由接口层直接落库为 ``cancelled``（终态），
执行线程的完成/失败落库均带守卫、不再能覆盖取消。服务启动时先做
崩溃恢复：上次 ``running`` 的作业标记为 ``interrupted``，``queued``
的作业继续排队执行；所有终态作业的未收尾条目一并收尾。
"""

from __future__ import annotations

import logging
import threading

from .jobs import run_match_job, run_spectrum_job
from .storage import Storage

log = logging.getLogger("seismic.scheduler")


class Scheduler:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        interrupted = self.storage.recover_interrupted()
        for jid in interrupted:
            log.warning("作业 %s 在进程重启时处于执行中，已标记为 interrupted", jid)
        self._thread = threading.Thread(
            target=self._loop, name="job-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def notify(self) -> None:
        """新作业入队后唤醒工作线程（避免等待轮询间隔）。"""

        self._wake.set()

    def _loop(self) -> None:
        runners = {
            "spectrum": run_spectrum_job,
            "match": run_match_job,
        }
        while not self._stop.is_set():
            job_id = self.storage.claim_next_queued()
            if job_id is None:
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue
            self._wake.clear()
            job = self.storage.get_job(job_id)
            try:
                result = runners[job["type"]](
                    job_id,
                    self.storage,
                    lambda j=job_id: self.storage.is_cancel_requested(j),
                )
                if result.get("cancelled"):
                    log.info("作业 %s 已取消", job_id)
                elif self.storage.complete_job(job_id, result):
                    log.info("作业 %s 完成", job_id)
                else:
                    # 取消请求在结果落库前一刻落库：取消优先，结果丢弃
                    log.info("作业 %s 在结果落库前已被取消，维持 cancelled", job_id)
            except Exception as exc:  # noqa: BLE001 - 兜底，作业级失败要落库
                log.exception("作业 %s 执行失败", job_id)
                if not self.storage.fail_job(job_id, f"作业执行失败：{exc}"):
                    log.info("作业 %s 已被取消，失败状态不再落库", job_id)

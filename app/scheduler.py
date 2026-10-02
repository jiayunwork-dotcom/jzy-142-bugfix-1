"""后台作业调度：单工作线程串行执行排队作业。

设计上刻意保持简单：一个 daemon 线程、一次只跑一个作业（反应谱计算
本身已在 NumPy 层并行，且 SQLite 单写），作业内部在条目之间、单条
记录的积分循环内（每 256 个时间步）以及匹配阶段前后轮询取消状态。

终态只由本线程在 :meth:`Storage.finalize_job` / :meth:`Storage.fail_job`
落定；取消接口只登记请求。服务启动时先做崩溃恢复：上次 ``running``
的作业标记为 ``interrupted``（已登记取消的仍判 cancelled），``queued``
的作业继续排队执行。
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

    def hard_kill(self) -> None:
        """模拟进程被 SIGKILL：不等待工作线程收尾，直接放弃引用。

        仅用于重启恢复测试——daemon 线程随解释句柄丢弃，作业在数据库里
        保持 running，下一个进程启动时走崩溃恢复。
        """

        self._stop.set()
        self._thread = None

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
                    final = self.storage.finalize_job(job_id, None)
                else:
                    final = self.storage.finalize_job(job_id, result)
                if final == "cancelled":
                    log.info("作业 %s 已取消（终态仲裁）", job_id)
                else:
                    log.info("作业 %s 完成", job_id)
            except Exception as exc:  # noqa: BLE001 - 兜底，作业级失败要落库
                log.exception("作业 %s 执行失败", job_id)
                self.storage.fail_job(job_id, f"作业执行失败：{exc}")

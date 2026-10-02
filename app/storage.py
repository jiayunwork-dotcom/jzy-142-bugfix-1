"""SQLite 持久化：记录、作业、逐条作业项。

库文件放在挂载卷（环境变量 ``SEISMIC_DB``，默认 ``/data/seismic.db``）。
采用 WAL 模式配合单进程 gunicorn + 后台调度线程；每个线程使用独立
连接（``check_same_thread=False`` + 线程内独占连接），写操作串行化。

作业状态机：``queued -> running -> completed / failed / cancelled``；
异常中断（进程退出）后重启时由 :meth:`Storage.recover_interrupted`
把 ``running`` 标记为 ``interrupted``，``queued`` 保持排队等待重跑。

取消与终态的约定（详见 README「取消、中断与条目终态」一节）：

- 取消请求方说了算：``request_cancel`` 在**同一事务**内把作业从
  ``queued/running`` 翻转为 ``cancelled`` 并收尾所有 ``pending`` 条目；
  此后 ``complete_job`` / ``fail_job`` 的守卫式 UPDATE
  （``WHERE status='running'``）不再生效，取消不会被任何收尾动作翻盘；
- 正在计算的条目允许算完并保留结果（``done``），未开始的条目一律
  收尾为 ``cancelled``；
- 作业进入任何终态后，作业项不得停留在 ``pending``/``running``：
  未跑完的条目按作业终态收尾为 ``cancelled`` / ``interrupted`` /
  ``aborted``（分别对应作业取消、进程中断、作业级失败）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    dt          REAL NOT NULL,
    npts        INTEGER NOT NULL,
    duration    REAL NOT NULL,
    unit_input  TEXT,
    fmt         TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    acc         BLOB NOT NULL,
    acc_dtype   TEXT NOT NULL DEFAULT 'float64',
    status      TEXT NOT NULL DEFAULT 'ready',
    error       TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL CHECK(type IN ('spectrum', 'match')),
    status      TEXT NOT NULL,
    params      TEXT NOT NULL DEFAULT '{}',
    progress    INTEGER NOT NULL DEFAULT 0,
    total       INTEGER NOT NULL DEFAULT 0,
    result      TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS job_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    record_id   TEXT REFERENCES records(id) ON DELETE SET NULL,
    seq         INTEGER NOT NULL,
    status      TEXT NOT NULL,
    result      TEXT,
    error       TEXT,
    UNIQUE(job_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_items_job ON job_items(job_id, seq);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

#: 作业终态 → 未跑完条目的收尾状态。``completed`` 不在其中：作业正常
#: 完成时所有条目必然已逐条落定（done/error/skipped），无需收尾。
JOB_TO_ITEM_TERMINAL = {
    "cancelled": "cancelled",      # 作业被取消：条目未运行
    "interrupted": "interrupted",  # 进程中断：条目未跑完
    "failed": "aborted",           # 作业级异常失败：条目未运行
}

#: 条目终态集合（文档与测试共用）：进入终态的作业，其条目状态必须落在这里。
ITEM_TERMINAL_STATES = frozenset(
    {"done", "error", "skipped"} | set(JOB_TO_ITEM_TERMINAL.values())
)

_INTERRUPTED_MSG = (
    "进程重启，作业在执行中中断；已算完的条目结果仍可查，"
    "整体结果不可用，请重新提交"
)


def _now() -> str:
    # 统一 UTC ISO8601，毫秒；仅用于展示，不参与结果内容
    import datetime as _dt

    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def json_dumps(obj: Any) -> str:
    """JSON 序列化，原生支持 NumPy 标量/数组，键排序保证确定性。"""

    return json.dumps(obj, default=_json_default, sort_keys=True, ensure_ascii=False)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"不可 JSON 序列化的类型：{type(o)!r}")


def _encode_acc(acc: np.ndarray) -> tuple[bytes, str]:
    a = np.ascontiguousarray(acc, dtype=np.float64)
    return a.tobytes(), str(a.dtype)


def _decode_acc(blob: bytes, dtype: str = "float64") -> np.ndarray:
    return np.frombuffer(blob, dtype=np.dtype(dtype)).copy()


@dataclass
class RecordRow:
    id: str
    name: str
    dt: float
    npts: int
    duration: float
    unit_input: str | None
    fmt: str
    metadata: dict
    acc: np.ndarray
    status: str
    error: str | None
    created_at: str


class Storage:
    """线程安全的 SQLite 封装（单进程使用）。"""

    def __init__(self, path: str) -> None:
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self._lock = threading.RLock()
        self._tls = threading.local()
        with self._lock:
            conn = self._connect()
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._tls, "conn", None)
        if c is None:
            c = self._connect()
            self._tls.conn = c
        return c

    # ---------------- 记录 ----------------

    def upsert_record(
        self,
        record_id: str,
        *,
        name: str,
        dt: float,
        unit_input: str | None,
        fmt: str,
        metadata: dict,
        acc: np.ndarray | None,
        status: str,
        error: str | None = None,
    ) -> None:
        blob, dtype = (b"", "float64") if acc is None else _encode_acc(acc)
        with self._lock:
            self.conn.execute(
                """INSERT INTO records(id, name, dt, npts, duration, unit_input,
                                      fmt, metadata, acc, acc_dtype, status,
                                      error, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     name=excluded.name, dt=excluded.dt, npts=excluded.npts,
                     duration=excluded.duration, unit_input=excluded.unit_input,
                     fmt=excluded.fmt, metadata=excluded.metadata,
                     acc=excluded.acc, acc_dtype=excluded.acc_dtype,
                     status=excluded.status, error=excluded.error""",
                (
                    record_id, name, float(dt), 0 if acc is None else acc.size,
                    0.0 if acc is None else float(dt * (acc.size - 1)),
                    unit_input, fmt, json_dumps(metadata or {}),
                    blob, dtype, status, error, _now(),
                ),
            )
            self.conn.commit()

    def get_record(self, record_id: str) -> RecordRow | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM records WHERE id=?", (record_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_records(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT id, name, dt, npts, duration, unit_input, fmt,
                          metadata, status, error, created_at
                   FROM records ORDER BY id"""
            ).fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "dt": r["dt"],
                "npts": r["npts"],
                "duration": r["duration"],
                "unit_input": r["unit_input"],
                "fmt": r["fmt"],
                "metadata": json.loads(r["metadata"]),
                "status": r["status"],
                "error": r["error"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def _row_to_record(self, row: sqlite3.Row) -> RecordRow:
        return RecordRow(
            id=row["id"],
            name=row["name"],
            dt=row["dt"],
            npts=row["npts"],
            duration=row["duration"],
            unit_input=row["unit_input"],
            fmt=row["fmt"],
            metadata=json.loads(row["metadata"]),
            acc=_decode_acc(row["acc"], row["acc_dtype"]) if row["acc"] else np.zeros(0),
            status=row["status"],
            error=row["error"],
            created_at=row["created_at"],
        )

    # ---------------- 作业 ----------------

    def create_job(
        self,
        job_id: str,
        job_type: str,
        params: dict,
        total: int,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO jobs(id, type, status, params, progress, total,
                                    created_at)
                   VALUES(?,?, 'queued', ?, 0, ?, ?)""",
                (job_id, job_type, json_dumps(params), int(total), _now()),
            )
            self.conn.commit()

    def add_job_items(self, job_id: str, entries: Iterable[tuple[str, int]]) -> None:
        """entries: (record_id 或 None, seq)。"""

        with self._lock:
            self.conn.executemany(
                """INSERT INTO job_items(job_id, record_id, seq, status)
                   VALUES(?, ?, ?, 'pending')""",
                [(job_id, rid, seq) for rid, seq in entries],
            )
            self.conn.commit()

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return self._job_dict(row) if row else None

    def get_job_params(self, job_id: str) -> dict:
        with self._lock:
            row = self.conn.execute(
                "SELECT params FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row["params"])

    def list_jobs(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._job_dict(r, with_result=False) for r in rows]

    def _job_dict(self, row: sqlite3.Row, *, with_result: bool = True) -> dict:
        d = {
            "id": row["id"],
            "type": row["type"],
            "status": row["status"],
            "params": json.loads(row["params"]),
            "progress": row["progress"],
            "total": row["total"],
            "error": row["error"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }
        if with_result:
            d["result"] = json.loads(row["result"]) if row["result"] else None
        return d

    def claim_next_queued(self) -> str | None:
        """原子领取一个排队作业并置为 running；无则返回 None。"""

        with self._lock:
            row = self.conn.execute(
                """SELECT id FROM jobs WHERE status='queued'
                   ORDER BY created_at ASC, id ASC LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            self.conn.execute(
                "UPDATE jobs SET status='running', started_at=? WHERE id=?",
                (_now(), row["id"]),
            )
            self.conn.commit()
            return row["id"]

    def request_cancel(self, job_id: str) -> bool:
        """请求取消：queued/running 在同一事务内原子翻转为 cancelled。

        同一事务里把所有 ``pending`` 条目收尾为 ``cancelled``；正在计算
        的条目（``running``）允许算完并保留结果，由执行线程（或进程死亡
        时的崩溃恢复）收尾。作业状态自此不可再被 complete/fail 改写。

        返回 True 表示本次请求把作业置为 cancelled；作业已处终态
        （完成/失败/取消/中断）时返回 False，调用方应回 409。
        """

        with self._lock:
            cur = self.conn.execute(
                """UPDATE jobs SET status='cancelled', finished_at=?
                   WHERE id=? AND status IN ('queued', 'running')""",
                (_now(), job_id),
            )
            if cur.rowcount == 0:
                return False
            self.conn.execute(
                """UPDATE job_items SET status='cancelled'
                   WHERE job_id=? AND status='pending'""",
                (job_id,),
            )
            self.conn.commit()
            return True

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT status FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return row is not None and row["status"] == "cancelled"

    def complete_job(self, job_id: str, result: dict) -> bool:
        """写入作业结果并置为 completed；作业已被取消时放弃写入。

        守卫 ``WHERE status='running'`` 保证与取消请求竞争时必有一方
        原子胜出：返回 False 表示取消先落库，作业保持 cancelled，
        本次汇总结果被丢弃（逐条结果仍可从作业项读取）。
        """

        with self._lock:
            cur = self.conn.execute(
                """UPDATE jobs SET status='completed', progress=total,
                                   result=?, finished_at=?
                   WHERE id=? AND status='running'""",
                (json_dumps(result), _now(), job_id),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def fail_job(self, job_id: str, error: str) -> bool:
        """置作业为 failed 并收尾未跑完条目；已被取消时维持 cancelled。

        无论本次是否抢到状态翻转，都会把终态作业遗留的
        pending/running 条目收尾（取消在前则收尾为 cancelled，
        否则为 aborted），保证终态作业没有未收尾条目。
        """

        with self._lock:
            cur = self.conn.execute(
                """UPDATE jobs SET status='failed', error=?, finished_at=?
                   WHERE id=? AND status='running'""",
                (error, _now(), job_id),
            )
            changed = cur.rowcount > 0
            self._finalize_unfinished_items_locked(job_id)
            self.conn.commit()
            return changed

    def finish_cancelled(self, job_id: str) -> None:
        """执行线程确认取消后的收尾：补齐可能遗留的未收尾条目。

        作业状态与 ``finished_at`` 已在 :meth:`request_cancel` 落库，
        这里只做防御性补齐（正常流程下没有遗留，是 no-op）。
        """

        with self._lock:
            self._finalize_unfinished_items_locked(job_id)
            self.conn.commit()

    def _finalize_unfinished_items_locked(self, job_id: str) -> int:
        """（调用方须已持锁）把终态作业的 pending/running 条目收尾为
        对应的条目终态（见 :data:`JOB_TO_ITEM_TERMINAL`），返回处理条数。"""

        row = self.conn.execute(
            "SELECT status FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None:
            return 0
        target = JOB_TO_ITEM_TERMINAL.get(row["status"])
        if target is None:
            return 0
        cur = self.conn.execute(
            """UPDATE job_items SET status=?
               WHERE job_id=? AND status IN ('pending', 'running')""",
            (target, job_id),
        )
        return cur.rowcount

    def set_job_progress(self, job_id: str, progress: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE jobs SET progress=? WHERE id=?", (int(progress), job_id)
            )
            self.conn.commit()

    def recover_interrupted(self) -> list[str]:
        """进程启动时调用：running -> interrupted，并收尾所有终态作业条目。

        崩溃可能发生在取消/失败收尾的中途，因此除 ``running`` 作业外，
        凡是已处终态（cancelled/interrupted/failed）的作业都要把遗留的
        pending/running 条目收尾为对应终态；已完成条目的结果保持可读。
        返回本次被标记为 interrupted 的作业 id。
        """

        with self._lock:
            rows = self.conn.execute(
                "SELECT id FROM jobs WHERE status='running'"
            ).fetchall()
            ids = [r["id"] for r in rows]
            self.conn.executemany(
                """UPDATE jobs SET status='interrupted', error=?, finished_at=?
                   WHERE id=?""",
                [(_INTERRUPTED_MSG, _now(), i) for i in ids],
            )
            rows = self.conn.execute(
                """SELECT id FROM jobs
                   WHERE status IN ('cancelled', 'interrupted', 'failed')"""
            ).fetchall()
            for r in rows:
                self._finalize_unfinished_items_locked(r["id"])
            self.conn.commit()
        return ids

    # ---------------- 作业项 ----------------

    def list_job_items(self, job_id: str) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT seq, record_id, status, result, error
                   FROM job_items WHERE job_id=? ORDER BY seq""",
                (job_id,),
            ).fetchall()
        return [
            {
                "seq": r["seq"],
                "record_id": r["record_id"],
                "status": r["status"],
                "result": json.loads(r["result"]) if r["result"] else None,
                "error": r["error"],
            }
            for r in rows
        ]

    def set_item_running(self, job_id: str, seq: int) -> bool:
        """把条目置为 running；条目已被取消收尾时返回 False。

        取消请求与执行线程之间存在检查-行动竞态：若取消先把该
        pending 条目收尾为 cancelled，本次更新不生效（返回 False，
        调用方应跳过该条），保证条目状态单调、cancelled 不被翻回。
        """

        with self._lock:
            cur = self.conn.execute(
                """UPDATE job_items SET status='running'
                   WHERE job_id=? AND seq=? AND status='pending'""",
                (job_id, seq),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def set_item_result(self, job_id: str, seq: int, result: dict) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE job_items SET status='done', result=? WHERE job_id=? AND seq=?",
                (json_dumps(result), job_id, seq),
            )
            self.conn.commit()

    def set_item_error(self, job_id: str, seq: int, error: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE job_items SET status='error', error=? WHERE job_id=? AND seq=?",
                (error, job_id, seq),
            )
            self.conn.commit()

    def set_item_skipped(self, job_id: str, seq: int, error: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE job_items SET status='skipped', error=? WHERE job_id=? AND seq=?",
                (error, job_id, seq),
            )
            self.conn.commit()

    def counts_by_status(self, job_id: str) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT status, COUNT(*) c FROM job_items WHERE job_id=? GROUP BY status",
                (job_id,),
            ).fetchall()
        counts = {r["status"]: r["c"] for r in rows}
        # 固定键集合，老调用方可以按键读取；新终态也一并给出
        for key in (
            "pending", "running", "done", "error", "skipped",
            "cancelled", "interrupted", "aborted",
        ):
            counts.setdefault(key, 0)
        return counts

    def close(self) -> None:
        c = getattr(self._tls, "conn", None)
        if c is not None:
            c.close()
            self._tls.conn = None

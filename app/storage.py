"""SQLite 持久化：记录、作业、逐条作业项。

库文件放在挂载卷（环境变量 ``SEISMIC_DB``，默认 ``/data/seismic.db``）。
采用 WAL 模式配合单进程 gunicorn + 后台调度线程；每个线程使用独立
连接（``check_same_thread=False`` + 线程内独占连接），写操作串行化。

作业状态机：``queued -> running -> completed / failed / cancelled``；
异常中断（进程退出）后重启时由 :meth:`Storage.recover_interrupted`
把 ``running`` 标记为 ``interrupted``，``queued`` 保持排队等待重跑。

取消与终态仲裁（修复「取消成功后又翻盘」的关键设计）
----------------------------------------------------
``jobs.cancel_requested`` 是一个与终态分离的**请求标志**：取消接口
只负责登记请求，绝不直接把 ``running`` 作业改写成终态。所有终态
（completed/failed/cancelled）只由执行线程在收尾时落定，落定动作在
一条 SQLite 事务里原子仲裁——取消标志先提交则判 cancelled，作业结果
先提交则判 completed，顺序由数据库单写串行化，因此任何一方都不可能
事后翻盘。作业项状态见 ``ITEM_TERMINAL_STATUSES``。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

# 作业项的非终态（作业进入终态时绝不允许残留）
ITEM_ACTIVE_STATUSES = ("pending", "running")
# 作业项的全部终态：
#   done/error/skipped —— 原有语义，老调用方均认识；
#   cancelled          —— 因作业取消而没跑成（未启动或算到一半被中止丢弃）；
#   interrupted        —— 因进程在执行中死亡而没跑完。
# 老调用方只按 done 取结果时天然忽略新状态；若按状态分支处理，应对
# 未知状态采取「终态、无结果」的保守解释（详见 README §8）。
ITEM_TERMINAL_STATUSES = ("done", "error", "skipped", "cancelled", "interrupted")
ITEM_ALL_STATUSES = ITEM_ACTIVE_STATUSES + ITEM_TERMINAL_STATUSES

# 作业取消原因写入作业项 error 列（该列对非 error 状态只是可读说明）
_REASON_CANCEL_PENDING = "作业取消，该条目未执行"
_REASON_CANCEL_ABORTED = "作业取消，该条目在计算中被中止，结果丢弃"
_REASON_INTERRUPTED_RUNNING = "进程在该条目执行中中断，结果不可用"
_REASON_INTERRUPTED_PENDING = "进程在作业执行中中断，该条目未执行"
_REASON_FAILED_PENDING = "作业整体失败，该条目未执行"

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
    cancel_requested INTEGER NOT NULL DEFAULT 0,
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
            self._migrate(conn)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """老库平滑升级：只新增列，不改动既有字段与接口路径。"""

        cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
        if "cancel_requested" not in cols:
            conn.execute(
                "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
            )

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
            "cancel_requested": bool(row["cancel_requested"]),
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

    def request_cancel(self, job_id: str) -> tuple[bool, str | None]:
        """登记取消请求，返回 ``(accepted, reason)``。

        - ``queued``：在同一条事务里直接判终态 cancelled（排队作业尚未被
          任何执行线程领取，不存在与收尾竞争），作业项全部置为 cancelled；
        - ``running``：只置 ``cancel_requested=1``，作业仍显示 running，
          终态由执行线程收尾时原子仲裁。这保证「接口说取消成功，最后就
          一定是 cancelled」——请求一旦落库，任何收尾都不能再写 completed；
        - 已在终态（completed/failed/cancelled/interrupted）：拒绝，调用方
          如实回 409。
        """

        with self._lock:
            row = self.conn.execute(
                "SELECT status, cancel_requested FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                return False, "not_found"
            if row["status"] in ("completed", "failed", "cancelled", "interrupted"):
                return False, row["status"]
            if row["cancel_requested"]:
                # 幂等：取消请求重复提交视为成功，且仍是同一承诺
                return True, row["status"]
            now = _now()
            if row["status"] == "queued":
                self.conn.execute(
                    """UPDATE jobs SET cancel_requested=1, status='cancelled',
                                      finished_at=? WHERE id=?""",
                    (now, job_id),
                )
                self.conn.execute(
                    """UPDATE job_items SET status='cancelled', error=?
                       WHERE job_id=? AND status IN ('pending','running')""",
                    (_REASON_CANCEL_PENDING, job_id),
                )
                self.conn.commit()
                return True, "cancelled"
            # running：只登记请求，不碰终态
            self.conn.execute(
                "UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,)
            )
            self.conn.commit()
            return True, "running"

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return row is not None and bool(row["cancel_requested"])

    def _finalize_active_items(self, job_id: str, *, cancelled: bool) -> None:
        """收尾时把仍处于非终态的作业项落到明确终态。

        cancelled=True（执行线程响应取消）：pending -> cancelled（没跑），
        running -> cancelled（算到一半，中止丢弃）。
        cancelled=False（作业失败）：pending -> skipped，running -> error。
        """

        if cancelled:
            self.conn.execute(
                """UPDATE job_items SET status='cancelled', error=?, result=NULL
                   WHERE job_id=? AND status='pending'""",
                (_REASON_CANCEL_PENDING, job_id),
            )
            self.conn.execute(
                """UPDATE job_items SET status='cancelled', error=?, result=NULL
                   WHERE job_id=? AND status='running'""",
                (_REASON_CANCEL_ABORTED, job_id),
            )
        else:
            self.conn.execute(
                """UPDATE job_items SET status='skipped', error=?
                   WHERE job_id=? AND status='pending'""",
                (_REASON_FAILED_PENDING, job_id),
            )
            self.conn.execute(
                """UPDATE job_items SET status='error', error=?
                   WHERE job_id=? AND status='running'""",
                ("作业执行失败，该条目未完成", job_id),
            )

    def finalize_job(self, job_id: str, result: dict | None) -> str:
        """执行线程收尾的**唯一**入口，在一条事务里原子仲裁终态。

        仲裁不依赖调用时序的巧合，而由 SQL 条件更新的命中行数决定：
        ``UPDATE ... WHERE cancel_requested=0 AND status='running'``
        若影响 0 行，说明取消标志已在本事务拿到写锁前提交（取消赢）；
        影响 1 行才是完成。因此即使不考虑进程内 RLock，结论也由数据库
        单写串行化保证。返回最终落定的状态。
        """

        payload = json_dumps(result) if result is not None else None
        with self._lock:
            row = self.conn.execute(
                "SELECT status FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                return "missing"
            if row["status"] in ("completed", "failed", "cancelled", "interrupted"):
                # 终态已落定（典型：崩溃恢复后的“僵尸”执行线程迟到一步），
                # 任何收尾都不得改写终态
                return row["status"]
            now = _now()
            cur = self.conn.execute(
                """UPDATE jobs SET status='completed', progress=total,
                                  result=?, finished_at=?
                   WHERE id=? AND status='running' AND cancel_requested=0""",
                (payload, now, job_id),
            )
            if cur.rowcount == 1:
                final = "completed"
            else:
                # 取消在本事务之前落库：丢弃结果，非终态作业项全部 cancelled
                self._finalize_active_items(job_id, cancelled=True)
                self.conn.execute(
                    """UPDATE jobs SET status='cancelled', progress=(
                         SELECT COUNT(*) FROM job_items
                         WHERE job_id=? AND status IN ('done','error','skipped')
                       ), finished_at=?
                       WHERE id=? AND status='running'""",
                    (job_id, now, job_id),
                )
                final = "cancelled"
            self.conn.commit()
        return final

    def fail_job(self, job_id: str, error: str) -> None:
        """执行线程异常兜底：若期间已登记取消，终态仍判给 cancelled。"""

        with self._lock:
            row = self.conn.execute(
                "SELECT status, cancel_requested FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                return
            if row["status"] in ("completed", "failed", "cancelled", "interrupted"):
                return  # 终态不可改写
            if row["cancel_requested"]:
                self._finalize_active_items(job_id, cancelled=True)
                self.conn.execute(
                    """UPDATE jobs SET status='cancelled', finished_at=?
                       WHERE id=? AND status='running'""",
                    (_now(), job_id),
                )
            else:
                self._finalize_active_items(job_id, cancelled=False)
                self.conn.execute(
                    """UPDATE jobs SET status='failed', error=?, finished_at=?
                       WHERE id=? AND status='running'""",
                    (error, _now(), job_id),
                )
            self.conn.commit()

    def set_job_progress(self, job_id: str, progress: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE jobs SET progress=? WHERE id=? AND status='running'",
                (int(progress), job_id),
            )
            self.conn.commit()

    def recover_interrupted(self) -> list[str]:
        """进程启动时调用，返回被标记为 interrupted 的作业 id。

        - ``running`` 且已有取消请求：终态判 cancelled（取消承诺不因进程
          死亡而失效；若崩溃发生在响应取消之前，由 item 上的说明区分）；
        - 其余 ``running``：interrupted。两种情况下作业项都要离开
          pending/running——running -> interrupted（算到一半进程死了），
          pending -> interrupted（没轮到）；已 done 的条目结果原样保留；
        - ``queued`` 保持不动，由新调度线程继续领取。
        """

        with self._lock:
            rows = self.conn.execute(
                "SELECT id, cancel_requested FROM jobs WHERE status='running'"
            ).fetchall()
            interrupted, cancelled = [], []
            for r in rows:
                (cancelled if r["cancel_requested"] else interrupted).append(r["id"])
            now = _now()
            if interrupted:
                qmarks = ",".join("?" for _ in interrupted)
                self.conn.execute(
                    f"""UPDATE job_items SET status='interrupted', error=?,
                        result=NULL
                        WHERE status='running' AND job_id IN ({qmarks})""",
                    (_REASON_INTERRUPTED_RUNNING, *interrupted),
                )
                self.conn.execute(
                    f"""UPDATE job_items SET status='interrupted', error=?
                        WHERE status='pending' AND job_id IN ({qmarks})""",
                    (_REASON_INTERRUPTED_PENDING, *interrupted),
                )
            if cancelled:
                qmarks = ",".join("?" for _ in cancelled)
                self.conn.execute(
                    f"""UPDATE job_items SET status='cancelled', error=?,
                        result=NULL
                        WHERE status='running' AND job_id IN ({qmarks})""",
                    ("取消已确认，进程在该条目执行期间重启，结果丢弃", *cancelled),
                )
                self.conn.execute(
                    f"""UPDATE job_items SET status='cancelled', error=?
                        WHERE status='pending' AND job_id IN ({qmarks})""",
                    ("取消已确认，进程重启前该条目未执行", *cancelled),
                )
            self.conn.executemany(
                """UPDATE jobs SET status='interrupted', error=?, finished_at=?
                   WHERE id=?""",
                [
                    ("进程重启，作业在执行中中断，结果不可用；请重新提交", now, i)
                    for i in interrupted
                ],
            )
            self.conn.executemany(
                """UPDATE jobs SET status='cancelled',
                   error=?, finished_at=? WHERE id=?""",
                [
                    ("取消已确认；进程在收尾落定前重启，作业仍按取消处理", now, i)
                    for i in cancelled
                ],
            )
            self.conn.commit()
        return interrupted

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

    def set_item_running(self, job_id: str, seq: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE job_items SET status='running' WHERE job_id=? AND seq=?",
                (job_id, seq),
            )
            self.conn.commit()

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

    def set_item_cancelled(self, job_id: str, seq: int) -> None:
        """正在算的条目响应取消而中止：结果丢弃，置 cancelled。"""

        with self._lock:
            self.conn.execute(
                """UPDATE job_items SET status='cancelled', result=NULL, error=?
                   WHERE job_id=? AND seq=?""",
                (_REASON_CANCEL_ABORTED, job_id, seq),
            )
            self.conn.commit()

    def counts_by_status(self, job_id: str) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT status, COUNT(*) c FROM job_items WHERE job_id=? GROUP BY status",
                (job_id,),
            ).fetchall()
        counts = {r["status"]: r["c"] for r in rows}
        for st in ITEM_ALL_STATUSES:
            counts.setdefault(st, 0)
        return counts

    def close(self) -> None:
        c = getattr(self._tls, "conn", None)
        if c is not None:
            c.close()
            self._tls.conn = None

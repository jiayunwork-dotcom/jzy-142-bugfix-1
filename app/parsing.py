"""强震记录文本解析。

支持两种常见两列文本格式及「单列 + 步长」格式：

1. 时间–加速度两列：``t a``，要求等步长，由数据本身核对一致性；
2. 加速度两列但第一列不是时间（例如序号–加速度）——通过步长一致性
   无法与时间列区分时，以 ``two_column_time`` / ``two_column_index``
   由调用方显式指定；自动模式下若首点接近 0、步长稳定则按时间列处理；
3. 单列加速度：必须由调用方给出 ``dt``。

以 ``#``、``%``、``*``、``!`` 开头的行以及空白分隔后首字段非数值的行
视为文件头注释，原文保留在 ``metadata.header`` 中；对若干常见关键字
（station/channel/component/dt/unit 等）额外抽取成结构化元数据。

解析严格逐行报错：空文件、非数值行、列数不一致、步长不一致都会抛出
:class:`~app.errors.RecordError`，消息中带行号，便于逐条定位。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .errors import RecordError

# 标准重力加速度，g -> m/s² 一律按此换算
GRAVITY = 9.80665

VALID_UNITS = ("m/s2", "g")

# 注释行首字符（PEER、PEER NGA、K-NET 等格式常用）
_COMMENT_PREFIXES = ("#", "%", "*", "!", "/")
_METADATA_KEYMAP = {
    "station": ("station", "台站", "site", "location"),
    "component": ("component", "comp", "orientation", "分量", "direction"),
    "channel": ("channel", "通道"),
    "network": ("network", "台网"),
    "event": ("event", "earthquake", "地震"),
    "dt": ("dt", "time step", "sampling interval", "步长"),
    "unit": ("unit", "units", "单位"),
    "name": ("name", "title", "记录名"),
}


@dataclass
class ParsedRecord:
    """解析结果。

    ``acc`` 的单位由 ``unit`` 声明，原样保留数值（不做单位换算），
    由后续加载步骤按声明单位统一换算到 m/s²。
    """

    acc: np.ndarray
    dt: float
    unit: str | None
    npts: int
    duration: float
    fmt: str
    metadata: dict = field(default_factory=dict)


def normalize_unit(unit: str | None) -> str | None:
    """把 ``m/s^2``、``M/S2``、``gal`` 等写法归一化。

    gal 不属于本服务接受的输入单位（量级容易搞错），直接报错。
    """

    if unit is None:
        return None
    u = unit.strip().lower().replace("^", "").replace("·", "")
    if u in ("g", "gal_g"):
        if u == "gal_g":
            return "g"
        return "g"
    if u in ("m/s2", "m/s/s", "ms-2", "meter/s2", "meter/s^2", "m sec-2"):
        return "m/s2"
    # 常见写法 "m/s^2" 去 ^ 后为 m/s2；cm/s2、gal 明确拒绝
    if u in ("cm/s2", "gal", "cm/s^2"):
        raise RecordError(
            f"不支持的加速度单位 '{unit}'：本服务仅接受 m/s2 或 g，"
            "请先将 gal/cm·s⁻² 记录换算后再上传"
        )
    raise RecordError(f"无法识别的加速度单位 '{unit}'：支持 m/s2 或 g")


def _looks_like_comment(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    return s.startswith(_COMMENT_PREFIXES)


def _extract_metadata(headers: list[str]) -> dict:
    meta: dict = {"header": headers}
    for line in headers:
        body = line.lstrip("".join(_COMMENT_PREFIXES)).strip()
        if not body:
            continue
        # key: value / key = value / key value
        m = re.match(r"^([A-Za-z一-鿿_/ ]+?)\s*[:=]\s*(.+)$", body)
        if not m:
            continue
        raw_key, raw_val = m.group(1).strip().lower(), m.group(2).strip()
        for canon, aliases in _METADATA_KEYMAP.items():
            if raw_key in aliases or any(a in raw_key for a in aliases):
                meta.setdefault(canon, raw_val)
                break
    return meta


def _parse_float(token: str, lineno: int) -> float:
    try:
        return float(token)
    except ValueError as exc:
        raise RecordError(
            f"第 {lineno} 行含有非数值数据 '{token}'，请检查文件内容或分隔符"
        ) from exc


def parse_record_text(
    text: str,
    *,
    filename: str = "<memory>",
    dt: float | None = None,
    unit: str | None = None,
    max_points: int = 200_000,
) -> ParsedRecord:
    """解析一条强震记录文本。

    参数
    ----
    text:
        文件文本内容（建议 UTF-8；调用方也可先做容错解码）。
    dt:
        单列格式必填的采样步长（秒）；两列格式下作为核对值（可选）。
    unit:
        调用方声明的单位，``'m/s2'`` 或 ``'g'``；为 ``None`` 时尝试从
        文件头 ``unit`` 字段推断，仍无法确定则交给后续步骤报错。
    """

    if text is None or not text.strip():
        raise RecordError(f"记录 {filename} 为空文件或全是空白")

    headers: list[str] = []
    rows: list[tuple[int, list[str]]] = []
    ncols: int | None = None

    for lineno, raw in enumerate(text.splitlines(), start=1):
        if _looks_like_comment(raw):
            if raw.strip():
                headers.append(raw.strip())
            continue
        parts = raw.replace(",", " ").replace(";", " ").split()
        if not parts:
            continue
        if ncols is None:
            ncols = len(parts)
        if len(parts) != ncols:
            raise RecordError(
                f"第 {lineno} 行有 {len(parts)} 列，与首个数据行的 {ncols} 列"
                "不一致，请确认分隔符与列格式"
            )
        rows.append((lineno, parts))

    if not rows:
        raise RecordError(f"记录 {filename} 中没有任何数值数据行")

    meta = _extract_metadata(headers)
    if unit is None and "unit" in meta:
        unit = meta["unit"]
    unit_norm = normalize_unit(unit)

    # 文件头里若写了 dt（例如 NGA 的 "DT= .0050 SEC"），单列时可采用
    header_dt = _metadata_dt(meta)

    if ncols == 1:
        use_dt = dt if dt is not None else header_dt
        if use_dt is None:
            raise RecordError(
                f"记录 {filename} 为单列加速度格式，必须显式提供采样步长 dt"
                "（秒），或在文件头写明 DT"
            )
        use_dt = _check_dt(use_dt, filename)
        acc = np.array(
            [_parse_float(parts[0], ln) for ln, parts in rows], dtype=np.float64
        )
        fmt = "single_column"
    elif ncols == 2:
        first = [_parse_float(parts[0], ln) for ln, parts in rows]
        second = np.array(
            [_parse_float(parts[1], ln) for ln, parts in rows], dtype=np.float64
        )
        f0, f1, f_last = first[0], first[1], first[-1]
        span = f_last - f0

        is_index = (
            dt is not None
            and _is_int_like(f0)
            and _is_int_like(f1)
            and abs((f1 - f0) - 1.0) < 1e-9
            and _is_int_like(span)
        )
        if is_index:
            use_dt = _check_dt(dt, filename)
            fmt = "two_column_index"
            acc = second
        else:
            # 按时间列处理：逐点核对等步长
            use_dt, ok, detail = _uniform_step(np.asarray(first, dtype=np.float64))
            if not ok:
                raise RecordError(
                    f"记录 {filename} 时间列步长不一致：名义步长 "
                    f"{detail['nominal']:.6g}s，但第 {detail['lineno']} 个采样"
                    f"（文件第 {detail['file_line']} 行）间隔为 "
                    f"{detail['actual']:.6g}s，相对偏差 "
                    f"{detail['rel']:.3e} 超过容差 1e-4"
                )
            if dt is not None and abs(use_dt - _check_dt(dt, filename)) > 1e-9:
                raise RecordError(
                    f"记录 {filename} 声明步长 {dt}s 与时间列步长 "
                    f"{use_dt:.6g}s 不一致"
                )
            fmt = "two_column_time"
            acc = second
    else:
        raise RecordError(
            f"记录 {filename} 首个数据行有 {ncols} 列：仅支持单列加速度或"
            "时间–加速度两列格式"
        )

    if not np.all(np.isfinite(acc)):
        bad = int(np.argmax(~np.isfinite(acc)))
        raise RecordError(
            f"记录 {filename} 第 {bad + 1} 个采样为 NaN 或 Inf，无法参与计算"
        )
    if len(acc) > max_points:
        raise RecordError(
            f"记录 {filename} 共 {len(acc)} 点，超过单条最多 {max_points} 点的限制"
        )
    if len(acc) < 2:
        raise RecordError(f"记录 {filename} 数据点不足（仅 {len(acc)} 点），至少需要 2 点")

    return ParsedRecord(
        acc=acc,
        dt=float(use_dt),
        unit=unit_norm,
        npts=len(acc),
        duration=float(use_dt * (len(acc) - 1)),
        fmt=fmt,
        metadata=meta,
    )


def _metadata_dt(meta: dict) -> float | None:
    if "dt" not in meta:
        return None
    raw = meta["dt"]
    m = re.search(r"[-+0-9.eE]+", raw)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _check_dt(dt: float, filename: str) -> float:
    try:
        dtf = float(dt)
    except (TypeError, ValueError) as exc:
        raise RecordError(f"记录 {filename} 的步长 dt 不是数值：{dt!r}") from exc
    if not np.isfinite(dtf) or dtf <= 0:
        raise RecordError(f"记录 {filename} 的步长 dt 必须为正数，收到 {dt!r}")
    return dtf


def _is_int_like(x: float) -> bool:
    return np.isfinite(x) and abs(x - round(x)) < 1e-9


def _uniform_step(t: np.ndarray) -> tuple[float, bool, dict]:
    """检查时间序列是否等步长，相对容差 1e-4（绝对容差 1e-9s）。"""

    nominal = (t[-1] - t[0]) / (len(t) - 1)
    tol = max(1e-9, 1e-4 * nominal)
    diffs = np.diff(t)
    bad = np.where(np.abs(diffs - nominal) > tol)[0]
    if bad.size == 0:
        if nominal <= 0:
            return nominal, False, {
                "nominal": nominal,
                "lineno": 1,
                "file_line": 1,
                "actual": nominal,
                "rel": float("inf"),
            }
        return float(nominal), True, {}
    k = int(bad[0])
    actual = float(diffs[k])
    rel = abs(actual - nominal) / nominal if nominal else float("inf")
    return float(nominal), False, {
        "nominal": float(nominal),
        # 数据段第 k 个间隔（0 起），文件行号无从得知时给数据序号
        "lineno": k + 1,
        "file_line": k + 2,
        "actual": actual,
        "rel": rel,
    }

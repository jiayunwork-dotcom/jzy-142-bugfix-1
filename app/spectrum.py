"""弹性反应谱计算。

对每条基线校正后的加速度记录、每个阻尼比，调用 :mod:`app.integrator`
对全部周期做 Newmark 递推，取相对位移 SD、相对速度 SV、绝对加速度 SA
的峰值，并派生伪速度 PSV = ω·SD、伪加速度 PSA = ω²·SD。

约定
----
- 周期列表默认在 0.02s–6s 之间对数等间距取 100 点（含端点）；
- 周期列表允许包含 0（零周期点）：该点 SA = PSA = PGA，
  SD = SV = PSV = 0；其余点走数值积分；
- 记录结束后统一补算 1.25 个（最长）自振周期的自由振动段并纳入峰值
  统计，规则见 :mod:`app.integrator`；
- 内部单位统一为 m/s²、m、m/s，另给以 g 计的 SA/PSA 便于对规范谱；
- PGA 取基线校正后加速度的最大绝对值，PGV/PGD 由两次梯形积分得到。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baseline import BASELINE_MODES, baseline_correct, ground_motion
from .errors import ValidationError
from .integrator import METHODS, newmark_response
from .parsing import GRAVITY

DEFAULT_TMIN = 0.02
DEFAULT_TMAX = 6.0
DEFAULT_NPTS = 100


def default_periods(n: int = DEFAULT_NPTS,
                    t_min: float = DEFAULT_TMIN,
                    t_max: float = DEFAULT_TMAX) -> np.ndarray:
    """周期对数等间距网格（默认 0.02–6s，100 点）。"""

    if n < 2:
        raise ValidationError("周期点数至少为 2")
    if not (0 < t_min < t_max):
        raise ValidationError(f"周期范围非法：需 0 < t_min < t_max，收到 {t_min}, {t_max}")
    return np.logspace(np.log10(t_min), np.log10(t_max), int(n))


def validate_periods(periods) -> np.ndarray:
    """校验周期列表：可转成有限数值、非负且严格递增。"""

    try:
        t = np.asarray(periods, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"周期列表无法转为数值数组：{exc}") from exc
    if t.ndim != 1 or t.size == 0:
        raise ValidationError("周期列表必须是非空的一维数组")
    if not np.all(np.isfinite(t)):
        raise ValidationError("周期列表中存在 NaN 或 Inf")
    if np.any(t < 0):
        raise ValidationError("周期列表不允许负值")
    if np.any(np.diff(t) <= 0):
        i = int(np.argmax(np.diff(t) <= 0))
        raise ValidationError(
            f"周期列表必须严格递增，第 {i + 1} 与第 {i + 2} 个点不满足"
            f"（{t[i]:g} >= {t[i + 1]:g}）"
        )
    return t


def validate_damping(zeta, *, allow_list: bool = True):
    """校验阻尼比：单个值或列表，均须落在 [0, 1]。"""

    if np.isscalar(zeta):
        z = float(zeta)
        if not np.isfinite(z) or not (0.0 <= z <= 1.0):
            raise ValidationError(f"阻尼比必须在 0 到 1 之间，收到 {zeta}")
        return z
    if not allow_list:
        raise ValidationError("此处只接受单个阻尼比")
    try:
        arr = [float(x) for x in zeta]
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"阻尼比列表无法转为数值：{exc}") from exc
    if not arr:
        raise ValidationError("阻尼比列表为空")
    for i, z in enumerate(arr):
        if not np.isfinite(z) or not (0.0 <= z <= 1.0):
            raise ValidationError(f"第 {i + 1} 个阻尼比 {z} 不在 0 到 1 之间")
    return arr


@dataclass
class SpectrumRequest:
    """单条记录的谱计算参数（存储层加载完记录后组装）。"""

    periods: np.ndarray
    dampings: tuple[float, ...]
    method: str = "average_acceleration"
    instability_policy: str = "refine"
    baseline_mode: str = "mean"


def prepare_spectrum_request(
    periods=None,
    dampings=0.05,
    method: str = "average_acceleration",
    instability_policy: str = "refine",
    baseline_mode: str = "mean",
) -> SpectrumRequest:
    """把接口层传入的参数做默认值填充与校验。"""

    if method not in METHODS:
        raise ValidationError(
            f"未知 Newmark 方法 '{method}'，可选：{', '.join(METHODS)}"
        )
    if instability_policy not in ("refine", "reject"):
        raise ValidationError(
            f"失稳策略 instability_policy 必须是 refine 或 reject，收到 "
            f"{instability_policy!r}"
        )
    if baseline_mode not in BASELINE_MODES:
        raise ValidationError(
            f"基线处理方式必须是 {', '.join(BASELINE_MODES)} 之一，收到 "
            f"{baseline_mode!r}"
        )
    t = default_periods() if periods is None else validate_periods(periods)
    zs = validate_damping(dampings, allow_list=True)
    if not isinstance(zs, list):
        zs = [zs]
    return SpectrumRequest(
        periods=t,
        dampings=tuple(zs),
        method=method,
        instability_policy=instability_policy,
        baseline_mode=baseline_mode,
    )


def compute_spectrum(
    acc_si: np.ndarray,
    dt: float,
    req: SpectrumRequest,
) -> dict:
    """对一条已换算为 m/s² 的加速度记录计算弹性反应谱。

    返回可直接 JSON 序列化（经 :func:`app.api.helpers.to_jsonable`）的 dict。
    """

    periods = req.periods
    corrected = baseline_correct(acc_si, dt, req.baseline_mode)
    vel, disp = ground_motion(corrected, dt)

    pga = float(np.max(np.abs(corrected)))
    pgv = float(np.max(np.abs(vel)))
    pgd = float(np.max(np.abs(disp)))

    pos = periods > 0
    t_pos = periods[pos]
    zero_idx = int(np.argmax(~pos)) if (~pos).any() else None

    spectra = []
    for zeta in req.dampings:
        sd = np.zeros(periods.size)
        sv = np.zeros(periods.size)
        sa = np.zeros(periods.size)
        res = newmark_response(
            corrected, dt, t_pos, zeta,
            method=req.method,
            instability_policy=req.instability_policy,
        )
        sd[pos] = res["sd"]
        sv[pos] = res["sv"]
        sa[pos] = res["sa"]
        if zero_idx is not None:
            sa[zero_idx] = pga  # 零周期点 SA = PGA

        omega = np.divide(
            2.0 * np.pi, periods,
            out=np.zeros_like(periods), where=pos,
        )
        psv = omega * sd
        psa = omega * omega * sd
        if zero_idx is not None:
            psa[zero_idx] = pga  # 零周期点 PSA = PGA

        unstable_full = np.zeros(periods.size, dtype=bool)
        unstable_full[pos] = res["unstable_mask"]

        spectra.append({
            "damping": zeta,
            "periods": periods.tolist(),
            "sd": sd.tolist(),
            "sv": sv.tolist(),
            "sa": sa.tolist(),
            "psv": psv.tolist(),
            "psa": psa.tolist(),
            "sa_g": (sa / GRAVITY).tolist(),
            "psa_g": (psa / GRAVITY).tolist(),
            "method": req.method,
            "dt_used": res["dt_used"],
            "refined": res["refined"],
            "refine_factor": res["refine_factor"],
            "unstable_at_input_dt": unstable_full.tolist(),
            "tail_steps": res["tail_steps"],
        })

    return {
        "periods": periods.tolist(),
        "dampings": list(req.dampings),
        "spectra": spectra,
        "ground_motion": {
            "pga": pga,
            "pgv": pgv,
            "pgd": pgd,
            "pga_g": pga / GRAVITY,
        },
        "baseline": req.baseline_mode,
        "free_vibration": {
            "included": True,
            "periods_factor": 1.25,
            "tail_duration": float(1.25 * periods.max()),
        },
        "units": {
            "time": "s",
            "sd": "m",
            "sv": "m/s",
            "psv": "m/s",
            "sa": "m/s2",
            "psa": "m/s2",
            "sa_g": "g",
            "psa_g": "g",
        },
    }

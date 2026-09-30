"""弹性反应谱与设计谱的匹配筛选。

判据
----
在指定周期区间 ``[t1, t2]`` 内，取记录谱周期点（并在端点补入区间
边界）作为统计网格，设计谱在该网格上按**对数–对数插值**取值
:math:`D_j`，记录谱（默认伪加速度 PSA）取 :math:`A_j`。对缩放后
记录 :math:`s A_j`，使对数误差平方和最小：

.. math::

    s^* = \\arg\\min_s \\sum_j [\\ln(s A_j) - \\ln D_j]^2
       = \\exp\\!\\left(\\frac1N \\sum_j \\ln\\frac{D_j}{A_j}\\right)

即各点比值的几何平均；缩放后的均方误差（自然对数空间）::

    mse = mean_j ( ln(s A_j / D_j) )²

缩放系数给定上下限 ``[s_min, s_max]``：若 ``s*`` 越限，记录按
``bounds_policy`` 处理——``"clamp"``（默认）取边界系数并按边界重算
误差后参与排序，``"reject"`` 直接剔除并在结果中注明原因与越限量。

排序按 mse 升序，取前 N 条；平均谱取这 N 条**缩放后谱的算术平均**
（ASCE 7 / GB 50011 选波的通行做法），另给几何平均供参考；并输出
平均谱与设计谱在设计谱全部控制点上的逐点比值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import ValidationError

SPECTRUM_QUANTITIES = ("psa", "sa")


@dataclass
class DesignSpectrum:
    """设计谱控制点（周期严格递增，谱值严格为正）。"""

    periods: np.ndarray
    values: np.ndarray
    unit: str = "g"

    def log_value_at(self, t: np.ndarray) -> np.ndarray:
        """在给定周期上做对数–对数插值；超出范围直接报错。"""

        t = np.asarray(t, dtype=np.float64)
        if np.any(t < self.periods[0] - 1e-12) or np.any(
            t > self.periods[-1] + 1e-12
        ):
            raise ValidationError(
                f"统计周期 {float(np.min(t)):g}–{float(np.max(t)):g}s 超出设计谱"
                f"定义范围 {self.periods[0]:g}–{self.periods[-1]:g}s，请调整"
                "匹配区间或补全设计谱"
            )
        return np.interp(
            np.log(t),
            np.log(self.periods),
            np.log(self.values),
        )


def make_design_spectrum(periods, values, unit: str = "g") -> DesignSpectrum:
    """从点表构造设计谱并做严格校验。"""

    try:
        tp = np.asarray(periods, dtype=np.float64)
        va = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"设计谱点表无法转为数值：{exc}") from exc
    if tp.ndim != 1 or tp.size < 2:
        raise ValidationError("设计谱点表至少需要 2 个控制点")
    if va.shape != tp.shape:
        raise ValidationError(
            f"设计谱周期 {tp.size} 点与谱值 {va.size} 点数量不一致"
        )
    if not (np.all(np.isfinite(tp)) and np.all(np.isfinite(va))):
        raise ValidationError("设计谱点表中存在 NaN 或 Inf")
    if np.any(tp <= 0) or np.any(va <= 0):
        raise ValidationError("设计谱的周期与谱值都必须严格为正（对数插值要求）")
    if np.any(np.diff(tp) <= 0):
        i = int(np.argmax(np.diff(tp) <= 0))
        raise ValidationError(
            f"设计谱周期必须严格递增，第 {i + 1}/{i + 2} 个点不满足"
        )
    return DesignSpectrum(periods=tp, values=va, unit=unit)


def _matching_grid(rec_periods: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """记录谱周期点 ∩ [t1,t2]，再补入区间端点，去重排序。"""

    grid = rec_periods[(rec_periods >= t1 - 1e-12) & (rec_periods <= t2 + 1e-12)]
    grid = np.concatenate([grid, [t1, t2]])
    grid = np.unique(grid)
    return grid[(grid >= t1 - 1e-12) & (grid <= t2 + 1e-12)]


def match_single(
    rec_periods: np.ndarray,
    rec_values: np.ndarray,
    design: DesignSpectrum,
    t1: float,
    t2: float,
    s_min: float = 0.5,
    s_max: float = 3.0,
    bounds_policy: str = "clamp",
) -> dict:
    """单条记录的最优缩放与误差。"""

    grid = _matching_grid(np.asarray(rec_periods, dtype=np.float64), t1, t2)
    if grid.size < 2:
        raise ValidationError("匹配区间内记录谱点数不足（至少需要区间端点）")
    a = np.exp(np.interp(np.log(grid), np.log(np.asarray(rec_periods)),
                         np.log(np.asarray(rec_values, dtype=np.float64))))
    if np.any(a <= 0):
        raise ValidationError("记录谱值必须为正才能做对数匹配")
    d = np.exp(design.log_value_at(grid))

    log_ratio = np.log(d) - np.log(a)
    s_star = float(math.exp(log_ratio.mean()))
    residual_star = log_ratio - math.log(s_star)
    mse_star = float(np.mean(residual_star ** 2))

    # 边界用 1e-12 相对容差，吸收 exp/log 往返的浮点误差
    tol = 1e-12 * max(1.0, abs(s_star))
    out_of_bounds = not (s_min - tol <= s_star <= s_max + tol)
    reason = None
    if out_of_bounds:
        if s_star < s_min:
            reason = f"最优缩放系数 {s_star:.4g} 低于下限 {s_min:g}"
        else:
            reason = f"最优缩放系数 {s_star:.4g} 高于上限 {s_max:g}"
        if bounds_policy == "reject":
            return {
                "scale_optimal": s_star,
                "scale": s_star,
                "mse_optimal": mse_star,
                "mse": mse_star,
                "out_of_bounds": True,
                "excluded": True,
                "reason": reason,
                "grid_periods": grid.tolist(),
            }
        s_used = float(min(max(s_star, s_min), s_max))
    else:
        s_used = s_star

    residual = np.log(a) + math.log(s_used) - np.log(d)
    mse = float(np.mean(residual ** 2))
    return {
        "scale_optimal": s_star,
        "scale": s_used,
        "mse_optimal": mse_star,
        "mse": mse,
        "rmse_log": float(math.sqrt(mse)),
        "out_of_bounds": out_of_bounds,
        "excluded": False,
        "reason": reason,
        "grid_periods": grid.tolist(),
        "grid_record": a.tolist(),
        "grid_design": d.tolist(),
    }


def match_batch(
    candidates: list[dict],
    design: DesignSpectrum,
    t1: float,
    t2: float,
    *,
    s_min: float = 0.5,
    s_max: float = 3.0,
    top_n: int = 7,
    quantity: str = "psa",
    bounds_policy: str = "clamp",
) -> dict:
    """对一批记录做匹配筛选。

    candidates 每条：``{"record_id", "periods", "psa"/"sa"}``，
    所有记录的周期网格应一致（本服务谱作业用同一周期列表，天然满足）。
    """

    if quantity not in SPECTRUM_QUANTITIES:
        raise ValidationError(
            f"匹配谱量 quantity 必须是 {SPECTRUM_QUANTITIES} 之一，收到 {quantity!r}"
        )
    if not candidates:
        raise ValidationError("没有可参与匹配的记录")
    if not (0 < t1 < t2):
        raise ValidationError(f"匹配周期区间非法：需 0 < t1 < t2，收到 {t1}, {t2}")
    if not (0 < s_min <= s_max):
        raise ValidationError(f"缩放系数上下限非法：需 0 < s_min <= s_max，收到 {s_min}, {s_max}")
    if top_n < 1:
        raise ValidationError("top_n 至少为 1")
    if t1 < design.periods[0] or t2 > design.periods[-1]:
        raise ValidationError(
            f"匹配区间 [{t1}, {t2}] 超出设计谱范围 "
            f"[{design.periods[0]}, {design.periods[-1]}]"
        )

    results = []
    for cand in candidates:
        r = match_single(
            cand["periods"], cand[quantity], design, t1, t2,
            s_min=s_min, s_max=s_max, bounds_policy=bounds_policy,
        )
        r["record_id"] = cand["record_id"]
        if "name" in cand:
            r["name"] = cand["name"]
        results.append(r)

    ranked = sorted(
        (r for r in results if not r["excluded"]),
        key=lambda r: (r["mse"], r["record_id"]),
    )
    excluded = [
        {
            "record_id": r["record_id"],
            **({"name": r["name"]} if "name" in r else {}),
            "scale_optimal": r["scale_optimal"],
            "reason": r["reason"],
        }
        for r in results
        if r["excluded"]
    ]
    chosen = ranked[: int(top_n)]

    # 平均谱（算术/几何）及与设计谱的逐点比值：网格取设计谱控制点
    avg = None
    geo = None
    ratios = None
    if chosen:
        periods0 = np.asarray(candidates[0]["periods"], dtype=np.float64)
        scaled = []
        for r in chosen:
            rec = next(c for c in candidates if c["record_id"] == r["record_id"])
            scaled.append(np.asarray(rec[quantity], dtype=np.float64) * r["scale"])
        scaled_arr = np.vstack(scaled)
        avg = np.mean(scaled_arr, axis=0)
        geo = np.exp(np.mean(np.log(scaled_arr), axis=0))
        knots = design.periods
        d_knots = design.values
        avg_at = np.exp(np.interp(np.log(knots), np.log(periods0), np.log(avg)))
        geo_at = np.exp(np.interp(np.log(knots), np.log(periods0), np.log(geo)))
        ratios = {
            "periods": knots.tolist(),
            "design": d_knots.tolist(),
            "average": avg_at.tolist(),
            "geometric_mean": geo_at.tolist(),
            "ratio_average_over_design": (avg_at / d_knots).tolist(),
            "ratio_geometric_over_design": (geo_at / d_knots).tolist(),
        }

    return {
        "config": {
            "t1": float(t1),
            "t2": float(t2),
            "s_min": float(s_min),
            "s_max": float(s_max),
            "top_n": int(top_n),
            "quantity": quantity,
            "bounds_policy": bounds_policy,
            "design_unit": design.unit,
        },
        "ranking": [
            {
                "rank": i + 1,
                "record_id": r["record_id"],
                **({"name": r["name"]} if "name" in r else {}),
                "scale": r["scale"],
                "scale_optimal": r["scale_optimal"],
                "mse": r["mse"],
                "rmse_log": r["rmse_log"],
                "out_of_bounds": r["out_of_bounds"],
            }
            for i, r in enumerate(chosen)
        ],
        "excluded": excluded,
        "average_spectrum": None if avg is None else {
            "periods": periods0.tolist(),
            "arithmetic_mean": avg.tolist(),
            "geometric_mean": geo.tolist(),
        },
        "ratios": ratios,
        "n_candidates": len(candidates),
        "n_excluded": len(excluded),
    }

"""基线处理与地面速度 / 位移积分。

为什么必须做基线处理
--------------------
强震仪得到的加速度记录常带有微小的零点偏移（直流分量）或缓慢漂移。
把这样的记录直接做两次梯形积分求地面位移时，恒定偏移 :math:`\\epsilon`
会产生随 :math:`t^2` 增长的虚假位移（:math:`\\tfrac12\\epsilon t^2`），
导致「长周期 SD 趋于地面峰值位移 PGD」这一物理关系完全失真。因此本
服务在计算反应谱之前，统一对加速度做基线校正，并把采用的方式原样写
入结果（``baseline`` 字段）与本文档。

三种方式
~~~~~~~~
- ``"none"``：不处理。仅在调用方确认记录已做过基线校正时使用；
- ``"mean"``（默认）：减去全程加速度均值，去除直流偏移。处理后地面
  速度在记录末端回到接近零；
- ``"linear"``：最小二乘拟合加速度的一次趋势线并扣除，同时去除直流
  与线性漂移，适合末端速度/位移明显不回零的记录。

地面速度、位移均用复合梯形积分、零初始条件递推；PGD 取位移序列的
最大绝对值。
"""

from __future__ import annotations

import numpy as np

from .errors import ValidationError

BASELINE_MODES = ("none", "mean", "linear")


def baseline_correct(acc: np.ndarray, dt: float, mode: str = "mean") -> np.ndarray:
    """按指定方式对加速度时程做基线校正，返回新数组（不修改输入）。"""

    if mode not in BASELINE_MODES:
        raise ValidationError(
            f"未知的基线处理方式 '{mode}'，可选：{', '.join(BASELINE_MODES)}"
        )
    a = np.asarray(acc, dtype=np.float64)
    if mode == "none":
        return a.copy()
    if mode == "mean":
        return a - a.mean()
    # linear：最小二乘直线去趋势
    n = a.size
    t = np.arange(n, dtype=np.float64) * dt
    # 用中心化时间做一次多项式拟合，数值上更稳
    tc = t - t.mean()
    slope = float(np.dot(tc, a - a.mean()) / np.dot(tc, tc))
    intercept = float(a.mean())
    trend = slope * tc + intercept
    return a - trend


def cumulative_trapezoid(y: np.ndarray, dt: float) -> np.ndarray:
    """复合梯形积分，零初始条件：返回长度与 y 相同、首点为 0 的原函数。"""

    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    out[0] = 0.0
    np.cumsum((y[:-1] + y[1:]) * 0.5 * dt, out=out[1:])
    return out


def ground_motion(
    acc: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """由基线校正后的加速度积分得到地面速度、地面位移（零初始条件）。"""

    vel = cumulative_trapezoid(acc, dt)
    disp = cumulative_trapezoid(vel, dt)
    return vel, disp

"""Newmark-β 逐步积分器（单自由度线弹性体系）。

运动方程（单位质量）::

    ẍ + 2ξω ẋ + ω² x = -a_g(t)

其中 x 为相对位移，a_g 为地面加速度。Newmark 递推：

.. math::

    x_{n+1} = x_n + Δt\\,v_n + Δt²[(\\tfrac12-β)a_n + β a_{n+1}]

    v_{n+1} = v_n + Δt[(1-γ)a_n + γ a_{n+1}]

本模块对「周期 × 阻尼比」的整个振荡器集合按时间步做向量化递推，
并在记录结束后补算自由振动段（地面加速度置零）。

稳定性与加密
------------
平均加速度法（γ=1/2, β=1/4）无条件稳定。线性加速度法（γ=1/2,
β=1/6）条件稳定，无阻尼临界条件 :math:`ωΔt ≤ √12`，即

.. math:: Δt/T ≤ √12/(2π) ≈ 0.55133。

若某个振荡器在原步长下越限，按 ``instability_policy``：

- ``"refine"``（默认）：把输入按线性插值整体加密到满足临界条件的整数
  倍数，再积分；线性加速度法本身假定区间内加速度线性变化，故线性
  插值加密与其假设严格一致，不引入额外近似；
- ``"reject"``：直接抛出 :class:`~app.errors.IntegrationError`，由
  调用方在结果/错误中注明。

结果中 ``refined``、``refine_factor``、``dt_used`` 以及逐周期的
``unstable_at_input_dt`` 掩码都会写清楚实际走的是哪条路径。
"""

from __future__ import annotations

import math

import numpy as np

from .errors import IntegrationError

METHODS: dict[str, tuple[float, float]] = {
    # name: (gamma, beta)
    "average_acceleration": (0.5, 0.25),
    "linear_acceleration": (0.5, 1.0 / 6.0),
}

# 线性加速度法稳定上限 dt/T = sqrt(12)/(2π)
LINEAR_DT_OVER_T_LIMIT = math.sqrt(12.0) / (2.0 * math.pi)

# 自由振动补算时长系数：补 1.25 个（最长）无阻尼自振周期。
# 无阻尼自由振动一个周期内必扫过 |x| 的理论最大值；有阻尼时峰值出现
# 得更早且包络指数衰减，1.25 周期对 0≤ξ≤1 均足够（详见 README）。
TAIL_PERIODS = 1.25


def required_subdivision(dt: float, t_min: float) -> int:
    """满足线性加速度法稳定条件所需的最小加密倍数（1 表示无需加密）。"""

    if t_min <= 0:
        return 1
    ratio = dt / t_min
    if ratio <= LINEAR_DT_OVER_T_LIMIT:
        return 1
    # 留 1e-12 的相对余量，避免恰好在临界点上的舍入误判
    m = math.ceil(ratio / LINEAR_DT_OVER_T_LIMIT * (1.0 + 1e-12))
    return max(2, int(m))


def refine_acceleration(
    acc: np.ndarray, dt: float, factor: int
) -> tuple[np.ndarray, float]:
    """线性插值把等步长加速度加密 ``factor`` 倍，返回 (新序列, 新步长)。"""

    if factor <= 1:
        return np.asarray(acc, dtype=np.float64), float(dt)
    n = acc.size
    old_index = np.arange(n, dtype=np.float64)
    new_index = np.linspace(0.0, n - 1, (n - 1) * factor + 1)
    acc_new = np.interp(new_index, old_index, acc).astype(np.float64)
    return acc_new, float(dt) / factor


def tail_step_count(t_max: float, dt_eff: float) -> int:
    """自由振动补算步数：至少 1 步，覆盖 1.25 个最长自振周期。"""

    return max(1, int(math.ceil(TAIL_PERIODS * t_max / dt_eff)))


def newmark_response(
    acc: np.ndarray,
    dt: float,
    periods: np.ndarray,
    damping: float,
    method: str = "average_acceleration",
    instability_policy: str = "refine",
) -> dict:
    """计算一批周期、单一阻尼比下的相对/绝对响应峰值。

    参数
    ----
    acc, dt:
        地面加速度（任意单位，输出谱同单位体系）与采样步长。
    periods:
        严格为正的周期序列（秒），T=0 点由上层单独处理。
    damping:
        阻尼比 ξ，允许 [0, 1]。

    返回
    ----
    dict，键：``sd`` / ``sv`` / ``sa``（长度与 periods 相同）、
    ``dt_used``、``refined``、``refine_factor``、``unstable_mask``、
    ``tail_steps``。
    """

    if method not in METHODS:
        raise IntegrationError(
            f"未知 Newmark 方法 '{method}'，可选：{', '.join(METHODS)}"
        )
    if instability_policy not in ("refine", "reject"):
        raise IntegrationError(
            f"未知失稳策略 '{instability_policy}'，可选 refine / reject"
        )
    periods = np.asarray(periods, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    t_min = float(periods.min())

    # 原始步长下逐周期的稳定性掩码（仅线性加速度法有意义）
    if method == "linear_acceleration":
        unstable_mask = (dt / periods) > LINEAR_DT_OVER_T_LIMIT
        factor = required_subdivision(dt, t_min) if unstable_mask.any() else 1
        if factor > 1:
            if instability_policy == "reject":
                bad_i = int(np.argmax(unstable_mask))
                raise IntegrationError(
                    f"线性加速度法在步长/周期比 Δt/T={dt / periods[bad_i]:.4f} "
                    f"超过稳定上限 {LINEAR_DT_OVER_T_LIMIT:.5f}（周期 "
                    f"{periods[bad_i]:.4g}s，Δt={dt:g}s）时失稳，策略为 "
                    "reject：请改用平均加速度法、加密输入或允许自动加密"
                )
            acc, dt_eff = refine_acceleration(acc, dt, factor)
        else:
            dt_eff, factor = float(dt), 1
        refined = factor > 1
    else:
        unstable_mask = np.zeros_like(periods, dtype=bool)
        dt_eff, factor, refined = float(dt), 1, False

    gamma, beta = METHODS[method]
    omega = 2.0 * np.pi / periods
    sd, sv, sa = _newmark_loop(
        acc, dt_eff, omega * omega, 2.0 * damping * omega, gamma, beta,
        t_max=float(periods.max()),
    )

    return {
        "sd": sd,
        "sv": sv,
        "sa": sa,
        "dt_used": dt_eff,
        "refined": refined,
        "refine_factor": factor,
        "unstable_mask": unstable_mask,
        "tail_steps": tail_step_count(float(periods.max()), dt_eff),
    }


def _newmark_loop(
    ag: np.ndarray,
    dt: float,
    w2: np.ndarray,
    c2xiw: np.ndarray,
    gamma: float,
    beta: float,
    t_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对所有振荡器（最后一维向量化）做强迫段 + 自由振动段递推。

    两套缓冲 (u0,v0,a0)/(u1,v1,a_new) 始终指向不同数组；每步把新状态
    算进第二套缓冲、统计峰值后成对交换引用。
    """

    a1 = 1.0 / (beta * dt * dt)     # 1/(βΔt²)
    a2 = 1.0 / (beta * dt)          # 1/(βΔt)
    a3 = 0.5 / beta - 1.0           # 1/(2β)-1
    c1 = gamma / (beta * dt)        # γ/(βΔt)
    c2 = gamma / beta - 1.0         # γ/β-1
    c3 = dt * (gamma / (2.0 * beta) - 1.0)
    one_m_gamma_dt = (1.0 - gamma) * dt
    gamma_dt = gamma * dt

    # 等效刚度（单位质量）：K̂ = ω² + 2ξω·c1 + a1
    keff = w2 + c2xiw * c1 + a1

    u = np.zeros_like(w2)
    v = np.zeros_like(w2)
    a = np.full_like(w2, -float(ag[0]))    # 静止起步：a(0) = -a_g(0)

    sd = np.zeros_like(w2)
    sv = np.zeros_like(w2)
    sa = np.zeros_like(w2)                  # t=0 绝对加速度恒为 0

    # 强迫段（ag[0] 已用于初值）步数 + 自由振动段步数，共用同一循环，
    # 自由段地面输入取 0。
    n_forced = ag.size - 1
    n_total = n_forced + tail_step_count(t_max, dt)

    for step in range(n_total):
        gk = float(ag[step + 1]) if step < n_forced else 0.0

        # 阻尼预测项 c1·u + c2·v + c3·a
        pred = c1 * u + c2 * v + c3 * a
        # 等效刚度移项后：
        # K̂·u_new = -gk + a1·u + a2·v + a3·a + 2ξω·pred
        # 注意 ω²·u_new 已并入 K̂，右端不再出现 ω²·u
        rhs = -gk + a1 * u + a2 * v + a3 * a + c2xiw * pred
        u_new = rhs / keff

        a_new = a1 * (u_new - u) - a2 * v - a3 * a
        v_new = v + one_m_gamma_dt * a + gamma_dt * a_new

        np.maximum(sd, np.abs(u_new), out=sd)
        np.maximum(sv, np.abs(v_new), out=sv)
        # 绝对加速度 = 相对加速度 + 地面加速度
        np.maximum(sa, np.abs(a_new + gk), out=sa)

        u, v, a = u_new, v_new, a_new

    return sd, sv, sa

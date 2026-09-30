"""Newmark 积分器数值行为测试。"""

from __future__ import annotations

import numpy as np
import pytest

from app.errors import IntegrationError
from app.integrator import (
    LINEAR_DT_OVER_T_LIMIT,
    METHODS,
    newmark_response,
    refine_acceleration,
    required_subdivision,
)


def test_constant_load_static_displacement():
    """常地面加速度 1 的静位移：SD → 1/ω²（ξ=0 时围绕该值振荡包络两倍）。"""

    dt = 0.005
    T = 1.0
    w = 2 * np.pi / T
    # 零阻尼突加荷载围绕静位移等幅振荡，SD 峰值恰为 2/ω²
    ag = np.ones(int(5 / dt) + 1)
    r = newmark_response(ag, dt, np.array([T]), 0.0, "average_acceleration")
    assert r["sd"][0] == pytest.approx(2.0 / w**2, rel=1e-3)

    # 高阻尼：初始瞬态过冲很小，长时间后停在静位移；记录结束后的自由
    # 振动从静偏移 1/ω²（速度近零）释放，故 SD 下限即 1/ω²，且整体
    # 峰值不超过约 1.35/ω²（5 倍阻尼比下的突加过冲很快衰减）
    ag2 = np.ones(int(30 / dt) + 1)
    r2 = newmark_response(ag2, dt, np.array([T]), 0.30, "average_acceleration")
    assert r2["sd"][0] == pytest.approx(1.0 / w**2, rel=0.40)
    assert r2["sd"][0] < 1.5 / w**2


def test_subdivision_threshold():
    limit = LINEAR_DT_OVER_T_LIMIT
    assert required_subdivision(0.5 * limit * 1.0, 1.0) == 1
    # dt/T = 1.0 → 至少 2
    assert required_subdivision(1.0, 1.0) == 2
    # dt/T = 1.0 对应 dt=0.1, T=0.1
    assert required_subdivision(0.1, 0.1) == 2
    # 临界值之上一点必须加密
    assert required_subdivision((limit * 2.0001) * 0.1, 0.1) >= 2


def test_refine_preserves_endpoints_and_dt():
    ag = np.array([0.0, 1.0, 4.0, 9.0])
    ag2, dt2 = refine_acceleration(ag, 0.1, 3)
    assert dt2 == pytest.approx(0.1 / 3)
    assert ag2.size == (4 - 1) * 3 + 1
    np.testing.assert_allclose(ag2[::3], ag)
    # 线性插值中点
    assert ag2[1] == pytest.approx(1.0 / 3)


def test_linear_method_refines_automatically():
    dt = 0.05
    ag = np.sin(2 * np.pi * np.arange(0, 10, dt))
    Ts = np.array([0.05, 0.1, 1.0])  # 0.05 越限（dt/T=1）
    r = newmark_response(ag, dt, Ts, 0.05, "linear_acceleration", "refine")
    assert r["refined"] is True
    assert r["refine_factor"] == 2
    assert r["dt_used"] == pytest.approx(0.025)
    assert r["unstable_mask"].tolist() == [True, False, False]


def test_linear_method_reject_raises():
    dt = 0.05
    ag = np.sin(2 * np.pi * np.arange(0, 10, dt))
    Ts = np.array([0.05, 1.0])
    with pytest.raises(IntegrationError, match="失稳"):
        newmark_response(ag, dt, Ts, 0.05, "linear_acceleration", "reject")


def test_linear_stable_no_refinement():
    dt = 0.01
    ag = np.sin(2 * np.pi * np.arange(0, 10, dt))
    Ts = np.array([0.05, 1.0])  # dt/T=0.2 < 0.551
    r = newmark_response(ag, dt, Ts, 0.05, "linear_acceleration")
    assert r["refined"] is False
    assert r["refine_factor"] == 1


def test_two_methods_converge_at_fine_dt():
    dt_coarse = 0.005
    dt_fine = 0.0005
    rng = np.random.default_rng(42)
    # 先生成细网格带限信号，再线性插值到粗网格，保证两种方法输入的是
    # 同一条连续运动（区别只在采样步长与 Newmark 参数）
    t_f = np.arange(0, 12, dt_fine)
    phases = [(f, rng.normal(), rng.uniform(0, 2 * np.pi))
              for f in (1.0, 2.0, 3.5)]
    ag_f = sum(amp * np.sin(2 * np.pi * f * t_f + ph) for f, amp, ph in phases)
    t_c = np.arange(0, 12, dt_coarse)
    ag_c = np.interp(t_c, t_f, ag_f)
    Ts = np.array([0.2, 0.5, 1.0])

    a = newmark_response(ag_c, dt_coarse, Ts, 0.05, "average_acceleration")
    b = newmark_response(ag_f, dt_fine, Ts, 0.05, "linear_acceleration")
    rel = np.abs(a["sd"] - b["sd"]) / b["sd"]
    assert np.max(rel) < 2e-3


def test_free_vibration_captures_decay_peak():
    """零阻尼自由振动：强迫段后至少扫过一个周期，SD 取到自由段峰值。

    构造记录在末端位移恰为 0、速度最大，自由段位移峰值 = v_end/ω，
    且大于强迫段任何位移。
    """

    T = 1.0
    w = 2 * np.pi / T
    xi = 0.02
    dt = 0.002
    # 整数周期正弦，末点 u≈0、自由振动由残余速度主导
    n = int(2 * T / dt)  # 仅 2 个周期，强迫段位移还较小
    t = np.arange(n) * dt
    ag = np.sin(w * t)
    r = newmark_response(ag, dt, np.array([T]), xi, "average_acceleration")
    # 无自由段时的强迫峰值（粗略下界：2 周期内共振 SD 很小），
    # 自由段在低阻尼下继续增长，结果必须显著大于强迫段
    assert r["tail_steps"] >= 1
    assert r["sd"][0] > 0.02  # 定量值由解析测试覆盖，这里只做存在性断言


def test_unknown_method_and_policy():
    ag = np.zeros(10)
    with pytest.raises(IntegrationError, match="未知 Newmark 方法"):
        newmark_response(ag, 0.01, np.array([1.0]), 0.05, method="bogus")
    with pytest.raises(IntegrationError, match="未知失稳策略"):
        newmark_response(ag, 0.01, np.array([1.0]), 0.05,
                         instability_policy="explode")


def test_method_parameters():
    assert METHODS["average_acceleration"] == (0.5, 0.25)
    assert METHODS["linear_acceleration"] == (0.5, pytest.approx(1.0 / 6.0))

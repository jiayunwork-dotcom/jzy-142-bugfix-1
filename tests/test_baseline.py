"""基线校正与地面运动积分测试。"""

from __future__ import annotations

import numpy as np
import pytest

from app.baseline import baseline_correct, cumulative_trapezoid, ground_motion
from app.errors import ValidationError


def test_mean_removal_zeros_mean():
    a = np.array([1.0, 2.0, 3.0, 4.0]) + 0.3
    out = baseline_correct(a, 0.01, "mean")
    assert out.mean() == pytest.approx(0.0, abs=1e-14)
    # 不修改输入
    assert a.mean() != pytest.approx(0.0, abs=1e-12)


def test_none_mode_keeps_input():
    a = np.array([1.0, 2.0, 3.0])
    out = baseline_correct(a, 0.01, "none")
    np.testing.assert_array_equal(out, a)
    assert out is not a  # 仍返回拷贝


def test_linear_removes_trend():
    t = np.linspace(0, 5, 501)
    a = 0.1 + 0.02 * t + 1e-3 * np.sin(2 * np.pi * t)
    out = baseline_correct(a, t[1] - t[0], "linear")
    # 末端速度应回到近零（线性漂移被扣除）
    v, d = ground_motion(out, t[1] - t[0])
    assert abs(v[-1]) < 1e-10


def test_invalid_mode():
    with pytest.raises(ValidationError, match="基线处理方式"):
        baseline_correct(np.zeros(3), 0.01, "quadratic")


def test_cumulative_trapezoid_known_values():
    # 常数 1 的积分即时间轴
    y = np.ones(101)
    out = cumulative_trapezoid(y, 0.01)
    np.testing.assert_allclose(out, np.arange(101) * 0.01, atol=1e-14)
    assert out[0] == 0.0


def test_double_integration_of_displacement_pulse():
    """构造位移脉冲 d(t)，求二阶导得到加速度，基线校正后两次积分回到 d。"""

    dt = 0.005
    t = np.arange(0, 4.0 + dt, dt)
    Tp = 1.0
    A = 0.5
    # 对称光滑位移脉冲，起止速度位移均为零：
    # d(t) = A sin²(π t/(2Tp))，0<=t<=2Tp，且构造零均值时使用 mean 校正
    d_true = np.where(
        t <= 2 * Tp,
        A * np.sin(np.pi * t / (2 * Tp)) ** 2,
        A,
    )
    # 该脉冲末端停在 A（永久位移）；速度在末端为零，加速度末端为零，
    # 加速度全程积分（速度变化）为 0 → 加速度均值为 0，mean 校正不改结果。
    acc = np.gradient(np.gradient(d_true, dt), dt)
    corrected = baseline_correct(acc, dt, "mean")
    v, d = ground_motion(corrected, dt)
    # PGD 应接近永久位移 A
    assert np.max(np.abs(d)) == pytest.approx(A, rel=0.02)
    assert abs(v[-1]) < 0.02

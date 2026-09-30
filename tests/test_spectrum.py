"""弹性反应谱计算与需求中列出的物理关系测试。"""

from __future__ import annotations

import numpy as np
import pytest

from app.errors import ValidationError
from app.parsing import GRAVITY
from app.spectrum import (
    compute_spectrum,
    default_periods,
    prepare_spectrum_request,
    validate_damping,
    validate_periods,
)


# --------------------------------------------------------------------------
# 正弦稳态放大解析解
#
# ẍ + 2ξωẋ + ω²x = -Ag sin Ωt，稳态伪加速度放大系数：
#   R = ω²U/Ag = 1 / sqrt((1-r²)² + (2ξr)²)，r = Ω/ω = T/Tg
#
# 为避免记录突停后自由振动峰值超过稳态幅值（共振附近可达 ~1.025U），
# 按解析稳态解选择截断相位 θ*，使自由振动初始包络 R0 =
# sqrt(U² + ((V0+ξωU)/ωd)²) 取到最小（恒 < U）。
# ---------------------------------------------------------------------------


def _end_phase(r: float, xi: float) -> float:
    """使自由振动初始包络 R0 最小的截断相位 θ = Ω·t_end（强迫 sin 相位）。

    稳态（取 Ag=1）::

        x(t) = ac·sinθ + bc·cosθ，ac = -(1-r²)/(Dω²)，bc = 2ξr/(Dω²)
        V(t) = Ω(ac·cosθ - bc·sinθ)，D = (1-r²)² + (2ξr)²

    自由振动初始包络 R0 = sqrt(x² + ((V+ξωx)/ωd)²)。把 R0² 写成关于
    (sinθ, cosθ) 的 2×2 二次型，最小特征值即 min R0（特征向量给 θ）。
    这样自动保证 R0 ≤ 稳态位移幅值 U = 1/(ω²√D)，记录突停后的自由段
    不会抬高峰值，数值谱即可直接对照稳态解析解。
    """

    wg = 2 * np.pi
    w = wg / r
    wd = w * np.sqrt(1 - xi**2)
    D = (1 - r * r) ** 2 + (2 * xi * r) ** 2
    ac = -(1 - r * r) / D / w**2
    bc = 2 * xi * r / D / w**2
    # 基向量 (sinθ, cosθ)
    xc = np.array([ac, bc])
    vc = np.array([-wg * bc, wg * ac])
    y = vc + xi * w * xc
    M = np.outer(xc, xc) + np.outer(y, y) / wd**2
    evals, evecs = np.linalg.eigh(M)
    z = evecs[:, int(np.argmin(evals))]
    return float(np.arctan2(z[0], z[1]) % (2 * np.pi))


# 为避免记录突加引发的**起振瞬态拍振**（其衰减时间尺度为 1/(ξω)，
# 在离共振点上会与强迫响应合拍，使前若干秒峰值高于稳态幅值），信号
# 用 20 个周期的半余弦包络平滑起步，再接稳态段；经验证 20 周期包络
# 可把残余瞬态压到稳态幅值的 1% 以内（r∈[0.5,1.5]）。
RAMP_CYCLES = 20
STEADY_CYCLES = 40


def _sine_record_at(r: float, xi: float, *, dt_frac=200):
    """以共振周期 Tg=1s 体系为参照，频率比 r=T/Tg 处的平滑起步正弦。

    末端再按 :func:`_end_phase` 给出的最优相位微补不到一个周期，使记录
    突停后的自由振动初始包络 R0 不超过稳态位移幅值 U。
    """

    Tg = 1.0
    T = r * Tg
    dt = Tg / dt_frac
    theta = _end_phase(r, xi)
    t_end = STEADY_CYCLES * Tg + theta / (2 * np.pi) * Tg
    t = np.arange(0, t_end + 0.5 * dt, dt)
    ramp = np.where(
        t < RAMP_CYCLES * Tg,
        0.5 * (1 - np.cos(np.pi * t / (RAMP_CYCLES * Tg))),
        1.0,
    )
    ag = ramp * np.sin(2 * np.pi * t / Tg)
    return ag, dt, T


def _analytic_factor(r: float, xi: float) -> float:
    return 1.0 / np.sqrt((1 - r * r) ** 2 + (2 * xi * r) ** 2)


@pytest.mark.parametrize("r", [0.7, 1.0, 1.3])
def test_sine_steady_state_amplification_within_1pct(r):
    xi = 0.05
    ag, dt, T = _sine_record_at(r, xi)
    req = prepare_spectrum_request(periods=[T], dampings=[xi])
    out = compute_spectrum(ag, dt, req)
    s = out["spectra"][0]
    psa = s["psa"][0]
    expected = _analytic_factor(r, xi)  # Ag = 1 m/s²
    rel = abs(psa - expected) / expected
    assert rel < 0.01, f"r={r}: PSA={psa:.5f}, 解析={expected:.5f}, 相对误差={rel:.4%}"
    # SA（绝对加速度峰值）在 r≈1 处与 PSA 同样接近解析值
    assert abs(s["sa"][0] - expected) / expected < 0.02


def test_sine_phase_trim_keeps_free_vibration_below_steady():
    """构造本身验证：最优相位处自由振动初始包络 R0 < 稳态幅值 U。"""

    xi = 0.05
    for r in (0.5, 0.7, 1.0, 1.3, 2.0):
        theta = _end_phase(r, xi)
        wg = 2 * np.pi
        w = wg / r
        wd = w * np.sqrt(1 - xi**2)
        D = (1 - r * r) ** 2 + (2 * xi * r) ** 2
        ac = -(1 - r * r) / D / w**2
        bc = 2 * xi * r / D / w**2
        x = ac * np.sin(theta) + bc * np.cos(theta)
        v = wg * (ac * np.cos(theta) - bc * np.sin(theta))
        r0 = np.sqrt(x**2 + ((v + xi * w * x) / wd) ** 2)
        u = 1 / (w**2 * np.sqrt(D))
        assert 0.0 <= theta < 2 * np.pi
        assert r0 < u


def test_scaling_linearity():
    """记录整体乘 k：所有谱值乘 k（谱形状不变）。

    用 ``baseline_mode='none'`` 隔离基线扣除的浮点舍入：积分器本身对
    输入是精确线性的，缩放后逐位一致。
    """

    rng = np.random.default_rng(0)
    dt = 0.01
    ag = rng.normal(size=3000) + 0.5 * np.sin(2 * np.pi * np.arange(3000) * dt / 0.8)
    Ts = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0])
    k = 2.7
    req1 = prepare_spectrum_request(periods=Ts, dampings=[0.05],
                                    baseline_mode="none")
    r1 = compute_spectrum(ag, dt, req1)
    r2 = compute_spectrum(k * ag, dt,
                          prepare_spectrum_request(periods=Ts, dampings=[0.05],
                                                   baseline_mode="none"))
    for qty in ("sd", "sv", "sa", "psv", "psa"):
        a = np.array(r1["spectra"][0][qty])
        b = np.array(r2["spectra"][0][qty])
        np.testing.assert_allclose(b, k * a, rtol=1e-10, err_msg=qty)
    assert r2["ground_motion"]["pga"] == pytest.approx(k * r1["ground_motion"]["pga"])


def test_scaling_linearity_with_mean_baseline():
    """默认 mean 基线下缩放关系同样成立（容差 1e-10，仅含求和舍入）。"""

    rng = np.random.default_rng(7)
    dt = 0.01
    ag = rng.normal(size=2000)
    Ts = np.array([0.1, 0.5, 1.0, 3.0])
    r1 = compute_spectrum(ag, dt,
                          prepare_spectrum_request(periods=Ts, dampings=[0.05]))
    r2 = compute_spectrum(3.0 * ag, dt,
                          prepare_spectrum_request(periods=Ts, dampings=[0.05]))
    np.testing.assert_allclose(r2["spectra"][0]["psa"],
                               3.0 * np.array(r1["spectra"][0]["psa"]), rtol=1e-10)


def test_zero_damping_sa_equals_psa():
    rng = np.random.default_rng(1)
    dt = 0.005
    t = np.arange(0, 15, dt)
    ag = rng.normal(scale=0.3, size=t.size) + np.sin(2 * np.pi * 1.5 * t)
    Ts = np.array([0.1, 0.3, 0.7, 1.0, 2.0, 4.0])
    out = compute_spectrum(ag, dt, prepare_spectrum_request(periods=Ts, dampings=[0.0]))
    s = out["spectra"][0]
    np.testing.assert_allclose(s["sa"], s["psa"], rtol=0, atol=1e-10)


def test_small_damping_sa_close_to_psa():
    """小阻尼中短周期段 SA 与 PSA 接近（多频带限信号，取统计包络）。"""

    rng = np.random.default_rng(2)
    dt = 0.005
    t = np.arange(0, 20, dt)
    ag = sum(
        rng.normal() * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        for f in (0.8, 1.2, 2.0, 3.0)
    )
    Ts = np.linspace(0.1, 1.0, 10)
    out = compute_spectrum(ag, dt, prepare_spectrum_request(periods=Ts, dampings=[0.01]))
    s = out["spectra"][0]
    rel = np.abs(np.array(s["sa"]) - np.array(s["psa"])) / np.array(s["psa"])
    assert np.median(rel) < 0.03


def test_short_period_sa_tends_to_pga():
    """周期趋小时 SA → PGA。

    用主频远低于振荡器自振频率的带限运动：随机噪声先低通平滑到
    ≲10Hz，再考察 250–500Hz 的振荡器（T=0.002–0.004s，dt=0.0002s）。
    注：含更高频成分的记录，SA 在短周期段高于 PGA 是物理现象（反应
    加速度包含体系自身高频反力），不构成对 T→0 极限的反例。
    """

    rng = np.random.default_rng(3)
    dt = 0.0001
    t = np.arange(0, 4, dt)
    # 明确带限于 1–4Hz 的多频正弦：PGA 附近波形光滑；取 250–500Hz 的
    # 振荡器（周期 0.002–0.004s），频率远高于激励，SA 应趋近 PGA。
    ag = sum(
        rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        for f in (1.0, 2.0, 3.5)
    )
    pga = np.abs(ag - ag.mean()).max()
    Ts = np.array([0.002, 0.003, 0.004])
    out = compute_spectrum(
        ag, dt, prepare_spectrum_request(periods=Ts, dampings=[0.05],
                                         baseline_mode="mean")
    )
    s = out["spectra"][0]
    ratios = np.array(s["sa"]) / pga
    assert abs(ratios[0] - 1.0) < 0.03
    assert abs(ratios[-1] - 1.0) < 0.10


def test_zero_period_point_is_pga():
    dt = 0.01
    ag = np.array([0.0, 0.5, -1.2, 0.3, 0.0] + [0.0] * 50)
    out = compute_spectrum(
        ag, dt, prepare_spectrum_request(periods=[0.0, 0.5, 1.0], dampings=[0.05])
    )
    s = out["spectra"][0]
    assert s["sa"][0] == pytest.approx(out["ground_motion"]["pga"])
    assert s["psa"][0] == pytest.approx(out["ground_motion"]["pga"])
    assert s["sd"][0] == 0.0
    assert s["sv"][0] == 0.0
    assert s["psv"][0] == 0.0


def test_long_period_sd_tends_to_pgd():
    """构造已知峰值位移的位移脉冲，长周期 SD 趋于 PGD。"""

    dt = 0.005
    t = np.arange(0, 4.0 + dt, dt)
    Tp = 1.0
    A = 0.5
    d_true = np.where(t <= 2 * Tp, A * np.sin(np.pi * t / (2 * Tp)) ** 2, A)
    acc = np.gradient(np.gradient(d_true, dt), dt)
    out = compute_spectrum(
        acc, dt,
        prepare_spectrum_request(periods=[20.0, 30.0, 40.0], dampings=[0.0],
                                 baseline_mode="mean"),
    )
    pgd = out["ground_motion"]["pgd"]
    assert pgd == pytest.approx(A, rel=0.02)
    s = out["spectra"][0]
    # T=40s = 40 Tp，体系近刚性跟随地面位移
    assert s["sd"][-1] == pytest.approx(A, rel=0.05)


def test_appending_zeros_does_not_change_spectrum():
    """记录末尾再补一段零，谱值不变（自由振动段已自动补算）。

    取 ``baseline_mode='none'``：mean 扣除在不同长度零段下得到的均值
    略有舍入差异，与「补零不变」这一积分器性质无关；自动补算的自由
    振动段保证多给零不会引入新峰值。
    """

    rng = np.random.default_rng(4)
    dt = 0.01
    ag = rng.normal(size=2000) + np.sin(2 * np.pi * np.arange(2000) * dt)
    Ts = np.array([0.05, 0.2, 0.5, 1.0, 2.0])
    req = prepare_spectrum_request(periods=Ts, dampings=[0.05],
                                   baseline_mode="none")
    r1 = compute_spectrum(ag, dt, req)
    r2 = compute_spectrum(np.concatenate([ag, np.zeros(500)]), dt,
                          prepare_spectrum_request(periods=Ts, dampings=[0.05],
                                                   baseline_mode="none"))
    for qty in ("sd", "sv", "sa", "psv", "psa"):
        np.testing.assert_allclose(
            r1["spectra"][0][qty], r2["spectra"][0][qty], atol=1e-10, err_msg=qty
        )


def test_default_periods_log_spaced():
    t = default_periods()
    assert t.size == 100
    assert t[0] == pytest.approx(0.02)
    assert t[-1] == pytest.approx(6.0)
    # 对数等间距
    logs = np.log(t)
    np.testing.assert_allclose(np.diff(logs), logs[1] - logs[0], rtol=1e-12)


def test_period_validation_messages():
    with pytest.raises(ValidationError, match="严格递增"):
        validate_periods([0.1, 0.2, 0.2, 0.4])
    with pytest.raises(ValidationError, match="不允许负值"):
        validate_periods([0.1, -0.2])
    with pytest.raises(ValidationError, match="NaN 或 Inf"):
        validate_periods([0.1, float("nan")])


def test_damping_validation():
    assert validate_damping(0.05) == 0.05
    assert validate_damping([0, 0.02, 0.05]) == [0, 0.02, 0.05]
    with pytest.raises(ValidationError, match="0 到 1"):
        validate_damping(1.2)
    with pytest.raises(ValidationError, match="第 2 个阻尼比"):
        validate_damping([0.05, -0.01])


def test_g_unit_conversion():
    """以 g 输入与换算成 m/s² 输入，谱值应一致（内部统一 SI）。"""

    dt = 0.01
    ag_si = np.sin(2 * np.pi * np.arange(0, 10, dt)) * 2.0
    ag_g = ag_si / GRAVITY
    Ts = np.array([0.3, 1.0, 2.0])
    r_si = compute_spectrum(ag_si, dt,
                            prepare_spectrum_request(periods=Ts, dampings=[0.05]))
    r_g = compute_spectrum(ag_g * GRAVITY, dt,
                           prepare_spectrum_request(periods=Ts, dampings=[0.05]))
    np.testing.assert_allclose(r_si["spectra"][0]["sa"],
                               r_g["spectra"][0]["sa"], rtol=1e-12)

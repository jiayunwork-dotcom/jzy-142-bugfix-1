"""谱匹配筛选测试：最优缩放、越限剔除、排序、平均谱与逐点比值。"""

from __future__ import annotations

import numpy as np
import pytest

from app.errors import ValidationError
from app.matching import make_design_spectrum, match_batch, match_single


def _flat_design(value=0.5, t=(0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 6.0)):
    return make_design_spectrum(list(t), [value] * len(t), unit="g")


def _flat_record(value, periods, rid, name=None):
    return {
        "record_id": rid,
        **({"name": name} if name else {}),
        "periods": list(periods),
        "psa": [value] * len(periods),
    }


def test_optimal_scale_is_geometric_mean_ratio():
    design = _flat_design(0.6)
    rec_periods = np.linspace(0.1, 1.0, 20)
    rec = _flat_record(0.2, rec_periods, "r1")
    r = match_single(rec_periods, rec["psa"], design, 0.1, 1.0)
    # 平坦谱：s* = D/A = 3
    assert r["scale"] == pytest.approx(3.0)
    assert r["mse"] == pytest.approx(0.0, abs=1e-12)
    assert r["out_of_bounds"] is False


def test_scaled_record_scales_inverse():
    """记录整体乘 k 后，最优缩放系数应除以 k（核心标度关系）。"""

    design = _flat_design(0.8)
    rec_periods = np.linspace(0.1, 2.0, 30)
    base = 0.4 * np.linspace(1.0, 1.5, rec_periods.size)
    r1 = match_single(rec_periods, base, design, 0.1, 2.0)
    k = 2.5
    r2 = match_single(rec_periods, k * base, design, 0.1, 2.0)
    assert r2["scale_optimal"] * k == pytest.approx(r1["scale_optimal"], rel=1e-12)
    # 缩放后对数误差不变
    assert r2["mse"] == pytest.approx(r1["mse"], abs=1e-12)


def test_mse_nonzero_for_mismatched_shape():
    tp = [0.1, 0.2, 0.5, 1.0, 2.0]
    design = make_design_spectrum(tp, [0.6, 0.6, 0.6, 0.4, 0.3], "g")
    rec_periods = np.linspace(0.1, 2.0, 25)
    values = np.full_like(rec_periods, 0.5)
    r = match_single(rec_periods, values, design, 0.1, 2.0,
                     s_min=0.01, s_max=100.0)
    assert r["mse"] > 0
    # 几何平均比值应在合理量级
    assert 0.5 < r["scale_optimal"] < 2.0


def test_out_of_bounds_clamp():
    design = _flat_design(0.6)
    periods = np.linspace(0.1, 1.0, 10)
    rec = _flat_record(0.1, periods, "r")  # 需要 s=6，超过上限 3
    r = match_single(periods, rec["psa"], design, 0.1, 1.0,
                     s_min=0.5, s_max=3.0, bounds_policy="clamp")
    assert r["out_of_bounds"] is True
    assert r["excluded"] is False
    assert r["scale"] == 3.0
    assert r["scale_optimal"] == pytest.approx(6.0)
    assert r["mse"] > r["mse_optimal"]
    assert "高于上限" in r["reason"]


def test_out_of_bounds_reject():
    design = _flat_design(0.6)
    periods = np.linspace(0.1, 1.0, 10)
    rec = _flat_record(0.1, periods, "r")
    r = match_single(periods, rec["psa"], design, 0.1, 1.0,
                     s_min=0.5, s_max=3.0, bounds_policy="reject")
    assert r["excluded"] is True
    assert r["out_of_bounds"] is True


def test_batch_ranking_and_top_n():
    design = _flat_design(0.5)
    periods = np.linspace(0.1, 1.0, 20)
    # 三条：精确匹配、需缩放但形状平坦、形状偏差大
    cands = [
        _flat_record(0.5, periods, "exact", "精确"),
        _flat_record(0.25, periods, "scale2", "两倍"),
        {
            "record_id": "shape",
            "name": "形状差",
            "periods": list(periods),
            "psa": list(0.5 * np.geomspace(0.2, 3.0, periods.size)),
        },
    ]
    out = match_batch(cands, design, 0.1, 1.0, s_min=0.5, s_max=3.0, top_n=2)
    assert [r["record_id"] for r in out["ranking"]] == ["exact", "scale2"]
    assert out["ranking"][0]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert out["ranking"][1]["scale"] == pytest.approx(2.0)
    assert len(out["ranking"]) == 2
    # 平均谱（缩放后）应贴近设计谱
    avg = np.array(out["average_spectrum"]["arithmetic_mean"])
    np.testing.assert_allclose(avg, 0.5, atol=2e-9)


def test_excluded_listed_with_reason():
    design = _flat_design(0.5)
    periods = np.linspace(0.1, 1.0, 10)
    cands = [
        _flat_record(0.5, periods, "ok"),
        _flat_record(0.05, periods, "needs10"),  # s*=10 > 3
    ]
    out = match_batch(cands, design, 0.1, 1.0, s_min=0.5, s_max=3.0,
                      top_n=5, bounds_policy="reject")
    assert len(out["excluded"]) == 1
    assert out["excluded"][0]["record_id"] == "needs10"
    assert "高于上限" in out["excluded"][0]["reason"]
    assert out["n_excluded"] == 1
    # 排序中不再包含被剔除记录
    assert all(r["record_id"] != "needs10" for r in out["ranking"])


def test_ratios_pointwise():
    tp = [0.1, 0.3, 1.0, 3.0]
    design = make_design_spectrum(tp, [0.4, 0.5, 0.6, 0.5], "g")
    periods = np.linspace(0.1, 3.0, 30)
    # 记录谱本身在全部 30 点上与设计谱一致（由对数插值定义）
    vals = np.exp(np.interp(np.log(periods), np.log(tp),
                            np.log([0.4, 0.5, 0.6, 0.5])))
    cands = [{"record_id": "a", "periods": list(periods), "psa": list(vals)}]
    out = match_batch(cands, design, 0.1, 3.0, s_min=0.1, s_max=10, top_n=1)
    ratios = out["ratios"]
    np.testing.assert_allclose(ratios["ratio_average_over_design"], 1.0, atol=1e-10)
    assert ratios["periods"] == tp


def test_design_spectrum_validation():
    with pytest.raises(ValidationError, match="至少需要 2 个"):
        make_design_spectrum([0.1], [0.5])
    with pytest.raises(ValidationError, match="严格递增"):
        make_design_spectrum([0.1, 0.1], [0.5, 0.6])
    with pytest.raises(ValidationError, match="严格为正"):
        make_design_spectrum([0.1, 0.2], [0.5, 0.0])
    with pytest.raises(ValidationError, match="数量不一致"):
        make_design_spectrum([0.1, 0.2], [0.5])


def test_interval_outside_design_range():
    design = _flat_design(0.5, t=(0.1, 1.0, 3.0))
    periods = np.linspace(0.1, 3.0, 10)
    with pytest.raises(ValidationError, match="超出设计谱范围"):
        match_batch([_flat_record(0.5, periods, "a")], design, 0.05, 3.0)


def test_invalid_bounds_and_topn():
    design = _flat_design(0.5)
    periods = np.linspace(0.1, 1.0, 10)
    with pytest.raises(ValidationError, match="s_min"):
        match_batch([_flat_record(0.5, periods, "a")], design, 0.1, 1.0,
                    s_min=2.0, s_max=1.0)
    with pytest.raises(ValidationError, match="top_n"):
        match_batch([_flat_record(0.5, periods, "a")], design, 0.1, 1.0, top_n=0)


def test_log_log_interpolation():
    design = make_design_spectrum([0.1, 1.0], [0.2, 0.8], "g")
    # 对数中点：sqrt(0.2*0.8)=0.4
    v = np.exp(design.log_value_at(np.array([np.sqrt(0.1)])))
    assert v[0] == pytest.approx(0.4)

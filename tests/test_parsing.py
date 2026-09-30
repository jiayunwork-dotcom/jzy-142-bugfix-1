"""记录解析与逐行错误信息测试。"""

from __future__ import annotations

import pytest

from app.errors import RecordError
from app.parsing import parse_record_text


def test_two_column_time():
    text = "\n".join(f"{i*0.01:.3f} {(i*0.01):.5f}" for i in range(101))
    rec = parse_record_text(text, filename="x.txt", unit="m/s2")
    assert rec.fmt == "two_column_time"
    assert rec.npts == 101
    assert rec.dt == pytest.approx(0.01)
    assert rec.unit == "m/s2"
    assert rec.duration == pytest.approx(1.0)


def test_single_column_requires_dt():
    text = "\n".join("0.1" for _ in range(10))
    with pytest.raises(RecordError, match="必须显式提供采样步长"):
        parse_record_text(text, filename="s.txt", unit="g")
    rec = parse_record_text(text, filename="s.txt", unit="g", dt=0.02)
    assert rec.fmt == "single_column"
    assert rec.dt == 0.02
    assert rec.unit == "g"


def test_index_column_with_dt():
    # 0,1,2,... 序号列 + 加速度
    text = "\n".join(f"{i} 0.01" for i in range(20))
    rec = parse_record_text(text, filename="i.txt", unit="m/s2", dt=0.005)
    assert rec.fmt == "two_column_index"
    assert rec.dt == 0.005
    assert rec.npts == 20


def test_empty_file():
    with pytest.raises(RecordError, match="空文件"):
        parse_record_text("   \n\t\n", filename="empty.txt")


def test_non_numeric_line_reports_line_number():
    text = "0.00 1.0\n0.01 2.0\n0.02 oops\n0.03 4.0\n"
    with pytest.raises(RecordError, match="第 3 行"):
        parse_record_text(text, filename="bad.txt", unit="m/s2")


def test_column_count_mismatch():
    text = "0.00 1.0 2.0\n0.01 3.0\n"
    with pytest.raises(RecordError, match="第 2 行有 2 列"):
        parse_record_text(text, filename="c.txt", unit="m/s2")


def test_uneven_time_step():
    lines = ["0.00 1.0", "0.01 1.1", "0.025 1.2", "0.035 1.3"]
    with pytest.raises(RecordError, match="步长不一致"):
        parse_record_text("\n".join(lines), filename="dt.txt", unit="m/s2")


def test_missing_unit_is_none_not_hard_error():
    text = "0.00 1.0\n0.01 1.1\n"
    rec = parse_record_text(text, filename="u.txt")
    assert rec.unit is None


def test_unknown_unit():
    with pytest.raises(RecordError, match="无法识别的加速度单位"):
        parse_record_text("0 1\n1 2\n", filename="u.txt", unit="ft/s2")


def test_gal_rejected():
    with pytest.raises(RecordError, match="仅接受 m/s2 或 g"):
        parse_record_text("0 1\n1 2\n", filename="u.txt", unit="gal")


def test_too_many_points():
    text = "".join(f"{i*0.01} 1\n" for i in range(5))
    with pytest.raises(RecordError, match="超过单条最多 3 点"):
        parse_record_text(text, filename="big.txt", unit="g", max_points=3)


def test_nan_inf_rejected():
    text = "0.00 1.0\n0.01 nan\n0.02 2.0\n"
    with pytest.raises(RecordError, match="NaN 或 Inf"):
        parse_record_text(text, filename="n.txt", unit="m/s2")


def test_header_metadata_preserved():
    text = (
        "# Station: CHY080\n"
        "# Component: N\n"
        "# DT: 0.005 SEC\n"
        "# 台站: 嘉义\n"
        "1 0.0\n2 0.1\n3 0.2\n"
    )
    rec = parse_record_text(text, filename="h.txt", unit="g", dt=0.005)
    assert rec.fmt == "two_column_index"
    assert rec.dt == pytest.approx(0.005)  # 显式步长与文件头一致
    assert rec.metadata["station"].lower() == "chy080"
    assert rec.metadata["component"] == "N"
    assert len(rec.metadata["header"]) == 4


def test_comment_prefixes_and_comma_separator():
    text = "% comment\n* another\n0.00,1.0\n0.01,2.0\n"
    rec = parse_record_text(text, filename="p.txt", unit="m/s2")
    assert rec.npts == 2
    assert len(rec.metadata["header"]) == 2


def test_declared_dt_mismatch_time_column():
    text = "0.00 1.0\n0.01 2.0\n0.02 3.0\n"
    with pytest.raises(RecordError, match="声明步长"):
        parse_record_text(text, filename="d.txt", unit="m/s2", dt=0.02)


def test_non_positive_dt():
    with pytest.raises(RecordError, match="步长 dt 必须为正数"):
        parse_record_text("1\n2\n", filename="d.txt", unit="g", dt=0)

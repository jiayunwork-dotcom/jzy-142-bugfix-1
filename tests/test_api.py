"""端到端 HTTP 接口测试：上传、建作业、进度、结果、取消、幂等/确定性。"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from tests.conftest import wait_for_status


def _two_col_text(dt=0.01, n=1000, seed=0, unit_amp=1.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    a = rng.normal(size=n) + np.sin(2 * np.pi * 2.0 * t)
    lines = [f"# Station: ST{seed}\n# Component: HNN\n"]
    lines += [f"{ti:.4f} {ai:.6f}\n" for ti, ai in zip(t, a * unit_amp)]
    return "".join(lines)


def _upload(client, text, name, unit="m/s2", dt=None):
    data = {"files": (io.BytesIO(text.encode()), name)}
    if unit:
        data["unit"] = unit
    if dt is not None:
        data["dt"] = str(dt)
    return client.post("/api/records", data=data,
                       content_type="multipart/form-data")


def test_health(client):
    rv = client.get("/health")
    assert rv.status_code == 200
    assert rv.get_json()["status"] == "ok"


def test_upload_and_list(client):
    rv = _upload(client, _two_col_text(), "r1.txt")
    assert rv.status_code == 200
    body = rv.get_json()
    assert body["ready"] == 1 and body["failed"] == 0
    rid = body["records"][0]["record_id"]
    assert body["records"][0]["metadata"]["station"] == "ST0"

    rv = client.get("/api/records")
    assert any(r["id"] == rid for r in rv.get_json()["records"])


def test_upload_json_single_column(client):
    dt = 0.005
    content = "\n".join(f"{x:.6f}" for x in np.sin(np.arange(500) * 0.1))
    rv = client.post("/api/records", json={
        "unit": "g", "dt": dt,
        "records": [{"name": "single.txt", "content": content}],
    })
    body = rv.get_json()
    assert body["ready"] == 1
    assert body["records"][0]["fmt"] == "single_column"
    assert body["records"][0]["unit"] == "g"


def test_upload_per_record_errors_do_not_fail_batch(client):
    good = _two_col_text(seed=1)
    bad_nonnumeric = "0.00 1.0\n0.01 oops\n"
    empty = "   \n"
    rv = client.post("/api/records", data={
        "files": [
            (io.BytesIO(good.encode()), "good.txt"),
            (io.BytesIO(bad_nonnumeric.encode()), "bad.txt"),
            (io.BytesIO(empty.encode()), "empty.txt"),
        ],
        "unit": "m/s2",
    }, content_type="multipart/form-data")
    body = rv.get_json()
    assert body["ready"] == 1
    assert body["failed"] == 2
    by_name = {r["name"]: r for r in body["records"]}
    assert "第 2 行" in by_name["bad.txt"]["error"]
    assert "空文件" in by_name["empty.txt"]["error"]
    # 失败记录也可列出
    listed = {r["id"]: r for r in client.get("/api/records").get_json()["records"]}
    assert by_name["bad.txt"]["record_id"] in listed
    assert listed[by_name["bad.txt"]["record_id"]]["status"] == "error"


def test_missing_unit_rejected_per_record(client):
    # JSON 批量中一条声明单位（成功）、一条不声明（该条失败，整批不挂）
    rv = client.post("/api/records", json={
        "records": [
            {"name": "withu.txt", "content": _two_col_text(seed=0),
             "unit": "m/s2"},
            {"name": "nou.txt", "content": _two_col_text(seed=1)},
        ],
    })
    body = rv.get_json()
    by_name = {r["name"]: r for r in body["records"]}
    assert by_name["withu.txt"]["status"] == "ready"
    assert by_name["nou.txt"]["status"] == "error"
    assert "单位" in by_name["nou.txt"]["error"]


def test_batch_over_200_rejected(client):
    content = json.dumps({
        "unit": "m/s2", "dt": 0.01,
        "records": [
            {"name": f"r{i}.txt", "content": "0.1\n0.2\n"}
            for i in range(201)
        ],
    })
    rv = client.post("/api/records", data=content,
                     content_type="application/json")
    assert rv.status_code == 400
    assert "200" in rv.get_json()["error"]


def test_bad_damping_rejected_at_create(client):
    up = _upload(client, _two_col_text(), "r.txt").get_json()
    rid = up["records"][0]["record_id"]
    rv = client.post("/api/jobs/spectrum", json={
        "record_ids": [rid], "dampings": [1.5], "periods": [0.1, 1.0],
    })
    assert rv.status_code == 400
    assert "0 到 1" in rv.get_json()["error"]


def test_nonincreasing_periods_rejected(client):
    up = _upload(client, _two_col_text(), "r.txt").get_json()
    rid = up["records"][0]["record_id"]
    rv = client.post("/api/jobs/spectrum", json={
        "record_ids": [rid], "periods": [0.5, 0.2],
    })
    assert rv.status_code == 400
    assert "严格递增" in rv.get_json()["error"]


def test_missing_record_id_rejected(client):
    rv = client.post("/api/jobs/spectrum",
                     json={"record_ids": ["rec_does_not_exist"],
                           "periods": [0.1, 1.0]})
    assert rv.status_code == 400
    assert "不存在" in rv.get_json()["error"]


@pytest.fixture()
def two_records(app_worker):
    client = app_worker.test_client()
    ids = []
    for seed in (0, 1):
        body = _upload(client, _two_col_text(seed=seed, n=800),
                       f"r{seed}.txt").get_json()
        ids.append(body["records"][0]["record_id"])
    return ids


def test_spectrum_job_end_to_end(app_worker, two_records):
    client = app_worker.test_client()
    rv = client.post("/api/jobs/spectrum", json={
        "record_ids": two_records,
        "periods": [0.0, 0.05, 0.1, 0.5, 1.0, 3.0],
        "dampings": [0.05],
    })
    assert rv.status_code == 201
    jid = rv.get_json()["job_id"]
    body = wait_for_status(client, jid)
    assert body["job"]["status"] == "completed"
    result = body["job"]["result"]
    assert result["summary"]["done"] == 2
    s0 = result["records"][0]["spectra"][0]
    # 零周期点 SA = PGA
    pga = result["records"][0]["ground_motion"]["pga"]
    assert s0["sa"][0] == pytest.approx(pga)
    assert s0["psa"][0] == pytest.approx(pga)
    # 有自由振动信息与方法标注
    assert result["records"][0]["free_vibration"]["included"] is True
    assert s0["method"] == "average_acceleration"

    # items 接口可查逐条结果
    items = client.get(f"/api/jobs/{jid}/items").get_json()
    assert items["counts"]["done"] == 2


def test_match_job_end_to_end(app_worker, two_records):
    client = app_worker.test_client()
    # 用记录在 5% 阻尼下的谱反推一个「设计谱」——平坦 0.3g，
    # 记录谱大致在该量级，保证匹配可解
    design = {"periods": [0.05, 0.1, 0.5, 1.0, 3.0],
              "values": [0.30, 0.30, 0.30, 0.30, 0.30], "unit": "g"}
    rv = client.post("/api/jobs/match", json={
        "record_ids": two_records,
        "periods": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0],
        "dampings": [0.05],
        "target_damping": 0.05,
        "design_spectrum": design,
        "t1": 0.1, "t2": 2.0,
        "s_min": 0.2, "s_max": 5.0, "top_n": 2,
        "quantity": "psa", "bounds_policy": "reject",
    })
    assert rv.status_code == 201
    jid = rv.get_json()["job_id"]
    body = wait_for_status(client, jid)
    assert body["job"]["status"] == "completed"
    m = body["job"]["result"]["matching"]
    assert 1 <= len(m["ranking"]) <= 2
    assert m["ratios"] is not None
    assert len(m["ratios"]["ratio_average_over_design"]) == 5


def test_match_invalid_design_spectrum(app_worker, two_records):
    client = app_worker.test_client()
    rv = client.post("/api/jobs/match", json={
        "record_ids": two_records,
        "design_spectrum": {"periods": [0.1, 0.05], "values": [0.3, 0.4]},
        "t1": 0.1, "t2": 0.05, "target_damping": 0.05,
    })
    assert rv.status_code == 400


def test_cancel_job_via_api(app_worker):
    client = app_worker.test_client()
    # 上传一条大记录
    big = _two_col_text(n=120000, seed=9)
    rid = _upload(client, big, "big.txt").get_json()["records"][0]["record_id"]
    rv = client.post("/api/jobs/spectrum", json={
        "record_ids": [rid],
        "periods": np.linspace(0.05, 6, 120).tolist(),
        "dampings": [0.05],
    })
    jid = rv.get_json()["job_id"]
    rv = client.post(f"/api/jobs/{jid}/cancel")
    assert rv.status_code == 200
    body = wait_for_status(client, jid)
    assert body["job"]["status"] in ("cancelled",)


def test_scaled_record_match_factor_divides_by_k(app_worker):
    """端到端：上传同一记录及其 k 倍（不同单位换算等价），匹配系数除以 k。"""

    client = app_worker.test_client()
    dt = 0.01
    t = np.arange(0, 10, dt)
    rng = np.random.default_rng(11)
    a = 0.4 * np.sin(2 * np.pi * 1.5 * t) + 0.2 * rng.normal(size=t.size)

    def upload(name, amp):
        content = "\n".join(f"{x:.6f}" for x in a * amp)
        return client.post("/api/records", json={
            "unit": "m/s2", "dt": dt,
            "records": [{"name": name, "content": content}],
        }).get_json()["records"][0]["record_id"]

    rid = upload("base.txt", 1.0)
    k = 2.0
    rid_k = upload("scaled.txt", k)

    design = {"periods": [0.05, 0.1, 0.5, 1.0, 3.0],
              "values": [0.10, 0.12, 0.12, 0.08, 0.04], "unit": "g"}

    def run_match(ids):
        jid = client.post("/api/jobs/match", json={
            "record_ids": ids,
            "periods": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0],
            "dampings": [0.05], "target_damping": 0.05,
            "design_spectrum": design,
            "t1": 0.1, "t2": 2.0,
            "s_min": 0.01, "s_max": 100.0, "top_n": 2,
        }).get_json()["job_id"]
        body = wait_for_status(client, jid)
        return body["job"]["result"]["matching"]["ranking"][0]["scale_optimal"]

    s = run_match([rid])
    s_k = run_match([rid_k])
    # 经上传/基线/积分整条数值链，1e-5 足以确认标度关系；
    # 匹配优化本身的精确标度律在 test_matching.py 以 1e-12 验证
    assert s_k * k == pytest.approx(s, rel=1e-5)


def test_repeat_submission_gives_identical_results(app_worker, two_records):
    """同一批记录、同样参数重复提交，结果（除作业 id/时间戳外）完全一致。"""

    client = app_worker.test_client()
    payload = {
        "record_ids": two_records,
        "periods": [0.05, 0.1, 0.3, 1.0, 3.0],
        "dampings": [0.02, 0.05],
    }
    ids = []
    for _ in range(2):
        jid = client.post("/api/jobs/spectrum", json=payload).get_json()["job_id"]
        ids.append(jid)
        wait_for_status(client, jid)
    results = []
    for jid in ids:
        job = client.get(f"/api/jobs/{jid}").get_json()["job"]
        results.append(job["result"])
    j1 = json.dumps(results[0], sort_keys=True)
    j2 = json.dumps(results[1], sort_keys=True)
    assert j1 == j2


def test_linear_method_reject_reports_in_record_result(app_worker):
    client = app_worker.test_client()
    # 单列 dt=0.1，最短周期 0.05 → dt/T=2 失稳，策略 reject
    content = "\n".join(f"{x:.6f}" for x in np.sin(np.arange(400) * 0.2))
    rid = client.post("/api/records", json={
        "unit": "m/s2", "dt": 0.1,
        "records": [{"name": "coarse.txt", "content": content}],
    }).get_json()["records"][0]["record_id"]
    jid = client.post("/api/jobs/spectrum", json={
        "record_ids": [rid],
        "periods": [0.05, 0.5, 2.0],
        "dampings": [0.05],
        "method": "linear_acceleration",
        "instability_policy": "reject",
    }).get_json()["job_id"]
    body = wait_for_status(client, jid)
    # 作业仍完成（整批不失败），单条记录标 error 并注明失稳
    assert body["job"]["status"] == "completed"
    rec = body["job"]["result"]["records"][0]
    assert rec["status"] == "error"
    assert "失稳" in rec["error"]


def test_linear_method_refine_reports_path(app_worker):
    client = app_worker.test_client()
    content = "\n".join(f"{x:.6f}" for x in np.sin(np.arange(400) * 0.2))
    rid = client.post("/api/records", json={
        "unit": "m/s2", "dt": 0.1,
        "records": [{"name": "coarse2.txt", "content": content}],
    }).get_json()["records"][0]["record_id"]
    jid = client.post("/api/jobs/spectrum", json={
        "record_ids": [rid],
        "periods": [0.05, 0.5, 2.0],
        "dampings": [0.05],
        "method": "linear_acceleration",
        "instability_policy": "refine",
    }).get_json()["job_id"]
    body = wait_for_status(client, jid)
    s = body["job"]["result"]["records"][0]["spectra"][0]
    assert s["refined"] is True
    assert s["refine_factor"] >= 2
    assert s["dt_used"] < 0.1
    assert s["unstable_at_input_dt"][0] is True
    assert s["unstable_at_input_dt"][1] is False

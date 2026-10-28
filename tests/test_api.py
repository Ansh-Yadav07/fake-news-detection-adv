import json
import time

import pytest

from conftest import GOOD_WIKI

LEGACY_KEYS = ["input_type", "transformer", "hybrid", "raw_features", "wikipedia",
               "verification", "timings", "total_time"]


def post(client, path, **kw):
    return client.post(path, **kw)


# ---- health ----
def test_health_returns_json_200(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.is_json and "status" in r.get_json()


# ---- /analyze validation ----
def test_analyze_missing_text_400(client):
    r = post(client, "/analyze", json={})
    assert r.status_code == 400 and r.is_json and "error" in r.get_json()


def test_analyze_non_string_text_400(client):
    r = post(client, "/analyze", json={"text": 123})
    assert r.status_code == 400 and r.is_json


def test_analyze_blank_text_400(client):
    r = post(client, "/analyze", json={"text": "   "})
    assert r.status_code == 400 and r.is_json


def test_analyze_malformed_json_400(client):
    r = post(client, "/analyze", data="notjson", content_type="application/json")
    assert r.status_code == 400 and r.is_json


def test_analyze_json_array_body_400(client):
    r = post(client, "/analyze", json=["text"])
    assert r.status_code == 400


def test_analyze_text_over_limit_413(client):
    r = post(client, "/analyze", json={"text": "a" * 10001})
    assert r.status_code == 413 and r.is_json


def test_analyze_text_at_limit_accepted(client):
    r = post(client, "/analyze", json={"text": "a" * 10000})
    assert r.status_code == 200


def test_analyze_oversized_body_413_json(client):
    r = post(client, "/analyze", data=json.dumps({"text": "a" * 70000}),
             content_type="application/json")
    assert r.status_code == 413 and r.is_json and r.get_json()["error"]


@pytest.mark.parametrize("route", ["/predict", "/verify"])
def test_legacy_routes_reject_non_string(client, route):
    assert post(client, route, json={"text": 5}).status_code == 400


@pytest.mark.parametrize("route", ["/predict", "/verify"])
def test_legacy_routes_reject_missing_text(client, route):
    assert post(client, route, json={}).status_code == 400


def test_unknown_route_404_json(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.is_json


def test_analyze_get_405_json(client):
    r = client.get("/analyze")
    assert r.status_code == 405 and r.is_json


# ---- /analyze happy path ----
def test_analyze_happy_path_has_legacy_and_new_keys(client, sample_text):
    r = post(client, "/analyze", json={"text": sample_text})
    d = r.get_json()
    assert r.status_code == 200
    for k in LEGACY_KEYS + ["task_status", "degraded", "failed_tasks"]:
        assert k in d, k
    assert d["degraded"] is False and d["failed_tasks"] == []
    assert set(d["task_status"]) == {"transformer", "hybrid", "linguistic", "wikipedia", "gnews"}
    assert all(v == "ok" for v in d["task_status"].values())
    assert d["transformer"]["label"] == "REAL"
    assert d["raw_features"]["word_count"] == len(sample_text.split())


# ---- budget ----
def test_global_budget_marks_slow_task_timeout(A, client, sample_text, monkeypatch, release_event):
    def slow(t, it=None):
        release_event.wait(30)
        return dict(GOOD_WIKI)

    monkeypatch.setattr(A, "_run_wikipedia_verification", slow)
    t0 = time.time()
    r = post(client, "/analyze", json={"text": sample_text})
    elapsed = time.time() - t0
    d = r.get_json()
    assert r.status_code == 200
    assert elapsed < A.ANALYZE_BUDGET_S + 1
    assert d["task_status"]["wikipedia"] == "timeout"
    assert d["degraded"] is True
    assert "wikipedia" in d["failed_tasks"]
    assert d["wikipedia"] is None


# ---- degradation ----
def test_failed_transformer_reports_unknown_and_hybrid_not_copied(A, client, sample_text, monkeypatch):
    monkeypatch.setattr(A, "_run_transformer", lambda t: None)
    d = post(client, "/analyze", json={"text": sample_text}).get_json()
    assert d["transformer"] == {"label": "UNKNOWN", "confidence": 0.5, "source": "none"}
    assert d["hybrid"]["label"] == "REAL"  # untouched, and not copied into transformer
    assert d["task_status"]["transformer"] == "error"
    assert d["degraded"] is True


def test_failed_hybrid_reports_unknown(A, client, sample_text, monkeypatch):
    monkeypatch.setattr(A, "_run_hybrid_ml", lambda t: None)
    d = post(client, "/analyze", json={"text": sample_text}).get_json()
    assert d["hybrid"] == {"label": "UNKNOWN", "confidence": 0.5}
    assert d["transformer"]["source"] == "local"


def test_task_raising_exception_is_error_not_crash(A, client, sample_text, monkeypatch):
    def boom(t, it):
        raise RuntimeError("x")
    monkeypatch.setattr(A, "_run_news_verification", boom)
    r = post(client, "/analyze", json={"text": sample_text})
    d = r.get_json()
    assert r.status_code == 200 and d["task_status"]["gnews"] == "error"


def test_all_tasks_failed_returns_503_generic(A, client, sample_text, monkeypatch):
    monkeypatch.setattr(A, "_run_transformer", lambda t: None)
    monkeypatch.setattr(A, "_run_hybrid_ml", lambda t: None)
    monkeypatch.setattr(A, "_run_wikipedia_verification", lambda t, it: None)
    monkeypatch.setattr(A, "_run_news_verification", lambda t, it: None)
    r = post(client, "/analyze", json={"text": sample_text})
    assert r.status_code == 503 and r.is_json
    assert "error" in r.get_json()


# ---- error leakage ----
def test_analyze_unhandled_exception_500_without_leak(A, client, sample_text, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("SECRET_INTERNAL_DETAIL")
    monkeypatch.setattr(A, "classify_input", boom)
    r = post(client, "/analyze", json={"text": sample_text})
    body = r.get_data(as_text=True)
    assert r.status_code == 500 and r.is_json
    assert "SECRET_INTERNAL_DETAIL" not in body
    assert r.get_json()["error"] == "Internal server error"


def test_predict_unhandled_exception_500_without_leak(A, client, sample_text, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("SECRET_INTERNAL_DETAIL")
    monkeypatch.setattr(A, "extract_features", boom)
    r = post(client, "/predict", json={"text": sample_text})
    assert r.status_code == 500
    assert "SECRET_INTERNAL_DETAIL" not in r.get_data(as_text=True)


def test_verify_exception_degrades_without_leak(A, client, sample_text, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("SECRET_INTERNAL_DETAIL")
    monkeypatch.setattr(A, "extract_main_claim", boom)
    r = post(client, "/verify", json={"text": sample_text})
    assert "SECRET_INTERNAL_DETAIL" not in r.get_data(as_text=True)
    assert r.get_json()["status"] == "UNVERIFIED"


def test_predict_happy_path(client, sample_text):
    r = post(client, "/predict", json={"text": sample_text})
    d = r.get_json()
    assert r.status_code == 200
    assert {"transformer", "hybrid", "raw_features"} <= set(d)


# ---- CORS ----
def _preflight(client, origin):
    return client.options("/analyze", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST"})


def test_cors_allowed_origin_gets_header(client):
    r = _preflight(client, "http://localhost:5173")
    assert r.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"


def test_cors_disallowed_origin_gets_no_header(client):
    r = _preflight(client, "http://evil.example")
    assert r.headers.get("Access-Control-Allow-Origin") is None

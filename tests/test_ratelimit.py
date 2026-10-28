"""Rate-limit tests use a private copy of app.py loaded with a tiny RATE_LIMIT,
because the limit string is bound when the module is imported."""
import importlib.util
import os

import pytest

from conftest import REPO_ROOT, SAMPLE_TEXT, GOOD_NEWS, GOOD_WIKI

LIMIT = 3


@pytest.fixture
def limited(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", f"{LIMIT} per minute")
    spec = importlib.util.spec_from_file_location("app_limited", os.path.join(REPO_ROOT, "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_http", None)
    monkeypatch.setattr(mod, "_run_transformer", lambda t: {"label": "REAL", "confidence": .9, "source": "local"})
    monkeypatch.setattr(mod, "_run_hybrid_ml", lambda t: {"label": "REAL", "confidence": .9})
    monkeypatch.setattr(mod, "_run_wikipedia_verification", lambda t, it: dict(GOOD_WIKI))
    monkeypatch.setattr(mod, "_run_news_verification", lambda t, it: dict(GOOD_NEWS))
    yield mod
    mod.limiter.reset()


def test_exceeding_limit_returns_429_json(limited):
    c = limited.app.test_client()
    codes = [c.post("/analyze", json={"text": SAMPLE_TEXT}).status_code for _ in range(LIMIT)]
    assert codes == [200] * LIMIT
    r = c.post("/analyze", json={"text": SAMPLE_TEXT})
    assert r.status_code == 429 and r.is_json and "error" in r.get_json()


def test_predict_is_rate_limited(limited):
    c = limited.app.test_client()
    codes = [c.post("/predict", json={"text": SAMPLE_TEXT}).status_code for _ in range(LIMIT)]
    assert codes == [200] * LIMIT
    r = c.post("/predict", json={"text": SAMPLE_TEXT})
    assert r.status_code == 429 and r.is_json


def test_limit_is_per_client_ip(limited):
    c = limited.app.test_client()
    for _ in range(LIMIT + 1):
        c.post("/analyze", json={"text": SAMPLE_TEXT}, headers={"X-Forwarded-For": "10.1.1.1"})
    r = c.post("/analyze", json={"text": SAMPLE_TEXT}, headers={"X-Forwarded-For": "10.2.2.2"})
    assert r.status_code == 200


def test_health_exempt_from_limit(limited):
    c = limited.app.test_client()
    assert all(c.get("/").status_code == 200 for _ in range(LIMIT * 5))

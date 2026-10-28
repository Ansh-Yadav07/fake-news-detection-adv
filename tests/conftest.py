"""Shared fixtures. Makes `import app` work without torch/transformers or network."""
import os
import sys
import types
import threading

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Environment must be set BEFORE app is imported (values are read at import) ---
# load_dotenv() never overrides variables that already exist, so setting these
# (even to "") keeps a developer's real .env keys out of the tests.
_TEST_ENV = {
    "RATE_LIMIT": "100000 per minute",
    "ANALYZE_BUDGET_S": "2",
    "ALLOWED_ORIGINS": "http://localhost:5173",
    "HF_TOKEN": "",
    "GNEWS_API_KEY": "",
    "NEWSDATA_API_KEY": "",
    "NEWSAPI_KEY": "",
    "SERPAPI_KEY": "",
}
os.environ.update(_TEST_ENV)

# --- Stub transformers so torch is never needed ---
_tf = types.ModuleType("transformers")
_tf.pipeline = lambda *a, **k: (lambda t: [{"label": "LABEL_1", "score": 0.9},
                                           {"label": "LABEL_0", "score": 0.1}])
sys.modules["transformers"] = _tf

# app.py uses repo-relative model paths
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import app as app_module  # noqa: E402

GOOD_WIKI = {"verification_score": 50, "status": "FOUND", "insights": []}
GOOD_NEWS = {"verification_score": 10, "status": "UNVERIFIED", "articles": [], "insights": []}
SAMPLE_TEXT = "Scientists announced a new study on climate today in the report."


class _NoNetwork:
    """Replacement for app._http: records and rejects every call."""

    def __init__(self):
        self.calls = []

    def _fail(self, method):
        def fn(*args, **kwargs):
            self.calls.append((method, args))
            raise AssertionError(f"Unexpected real HTTP {method} in test")
        return fn

    def __getattr__(self, name):
        if name in ("get", "post", "put", "delete", "request", "head"):
            return self._fail(name)
        raise AttributeError(name)


@pytest.fixture(scope="session")
def A():
    return app_module


@pytest.fixture(autouse=True)
def isolate(monkeypatch, A):
    """Per-test: reset limiter + caches, block network, stub every external task."""
    A.limiter.reset()
    with A._cache_lock:
        A._cache.clear()

    net = _NoNetwork()
    monkeypatch.setattr(A, "_http", net)
    for key in ("GNEWS_API_KEY", "NEWSDATA_API_KEY", "NEWSAPI_KEY", "SERPAPI_KEY", "HF_TOKEN"):
        monkeypatch.setattr(A, key, "")

    monkeypatch.setattr(A, "search_wikipedia", lambda q, timeout=1.5: None)
    monkeypatch.setattr(A, "_run_transformer",
                        lambda t: {"label": "REAL", "confidence": 0.9, "source": "local"})
    monkeypatch.setattr(A, "_run_hybrid_ml", lambda t: {"label": "REAL", "confidence": 0.8})
    monkeypatch.setattr(A, "_run_wikipedia_verification", lambda t, it: dict(GOOD_WIKI))
    monkeypatch.setattr(A, "_run_news_verification", lambda t, it: dict(GOOD_NEWS))
    monkeypatch.setattr(A, "ANALYZE_BUDGET_S", 2.0)

    yield net

    assert net.calls == [], f"Test attempted real network calls: {net.calls}"
    A.limiter.reset()


@pytest.fixture
def client(A):
    return A.app.test_client()


@pytest.fixture
def release_event():
    """Event for slow fake tasks; always set at teardown so pool threads never linger."""
    ev = threading.Event()
    yield ev
    ev.set()


@pytest.fixture
def sample_text():
    return SAMPLE_TEXT

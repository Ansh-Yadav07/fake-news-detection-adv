import pytest

from conftest import SAMPLE_TEXT


# ---- _normalize_label ----
@pytest.mark.parametrize("raw,expected", [
    ("REAL", "REAL"), ("real", "REAL"), ("Real", "REAL"), ("LABEL_1", "REAL"), ("1", "REAL"), (1, "REAL"),
    ("FAKE", "FAKE"), ("fake", "FAKE"), ("LABEL_0", "FAKE"), ("0", "FAKE"), (0, "FAKE"),
    ("weird", "UNKNOWN"), (None, "UNKNOWN"), ("", "UNKNOWN"), ("LABEL_2", "UNKNOWN"),
])
def test_normalize_label(A, raw, expected):
    assert A._normalize_label(raw) == expected


# ---- _safe_err ----
def test_safe_err_redacts_query_string_keys(A):
    msg = A._safe_err(Exception("GET https://a?apikey=SECRETKEY123&q=1 failed"))
    assert "SECRETKEY123" not in msg and "apikey=***" in msg


def test_safe_err_redacts_configured_secret_anywhere(A, monkeypatch):
    monkeypatch.setattr(A, "GNEWS_API_KEY", "TOPSECRETVALUE")
    assert "TOPSECRETVALUE" not in A._safe_err(ValueError("bad header TOPSECRETVALUE here"))


def test_safe_err_includes_type_and_truncates(A):
    out = A._safe_err(ValueError("x" * 1000))
    assert out.startswith("ValueError: ") and len(out) <= 200 + len("ValueError: ")


# ---- news cache ----
def _article(title="T"):
    return {"title": title, "description": "", "source": "s", "url": "u", "published_at": ""}


def test_news_second_identical_call_hits_cache(A, monkeypatch):
    calls = {"n": 0}

    def ok(q, m=10, timeout=2.0):
        calls["n"] += 1
        return [_article()]
    monkeypatch.setattr(A, "search_gnews", ok)
    monkeypatch.setattr(A, "search_newsdata", lambda *a, **k: [])
    monkeypatch.setattr(A, "search_newsapi", lambda *a, **k: [])
    first = A.search_all_news_sources("Climate Summit Announced")
    second = A.search_all_news_sources("climate summit announced")
    assert calls["n"] == 1 and first == second


def test_news_failed_lookup_not_cached(A, monkeypatch):
    monkeypatch.setattr(A, "search_gnews", lambda *a, **k: None)
    monkeypatch.setattr(A, "search_newsdata", lambda *a, **k: [])
    monkeypatch.setattr(A, "search_newsapi", lambda *a, **k: [])
    q = "other query words here"
    assert A.search_all_news_sources(q) == []
    assert A._cache_get(("news", A._clean_search_query(q).lower(), 10)) is A._MISS


# ---- classify_input ----
def test_classify_short_plain_fact_is_fact_claim(A):
    assert A.classify_input("the sky looks blue") == "fact_claim"


def test_classify_strong_news_signals_is_news_article(A):
    assert A.classify_input("USTR proposes new tariffs on Chinese imports") == "news_article"


def test_classify_short_with_one_signal_is_mixed(A):
    assert A.classify_input("the market is quiet") == "mixed"


def test_classify_long_with_event_keyword_is_news_article(A):
    assert A.classify_input(" ".join(["word"] * 30) + " election") == "news_article"


def test_classify_long_without_keywords_is_mixed(A):
    assert A.classify_input(" ".join(["lorem"] * 30)) == "mixed"


# ---- _clean_search_query ----
def test_clean_query_strips_punctuation_stopwords_and_short_words(A):
    assert A._clean_search_query("The kitten, and the cat Puppy!") == "kitten Puppy"


def test_clean_query_caps_at_six_words_prioritising_capitalised(A):
    out = A._clean_search_query("alpha1 bravo1 charlie1 delta1 echo11 foxtrot1 Zulu Yankee")
    words = out.split()
    assert len(words) == 6 and words[:2] == ["Zulu", "Yankee"]


def test_clean_query_falls_back_to_original_words_if_all_filtered(A):
    assert A._clean_search_query("an if it") == "an if it"


# ---- is_trusted_source ----
@pytest.mark.parametrize("name", [
    "Reuters", "BBC News", "  The Hindu ", "NDTV", "Associated Press",
    "apnews.com", "www.reuters.com", "timesofindia", "BBC-Sport", "Hindustan Times",
])
def test_is_trusted_source_known(A, name):
    assert A.is_trusted_source(name) is True


@pytest.mark.parametrize("name", ["Random Blog", "FakeNewsDaily.biz", "Fox News", "The Times", "AP"])
def test_is_trusted_source_unknown(A, name):
    assert A.is_trusted_source(name) is False


@pytest.mark.parametrize("name", ["", "   ", "News", "The", "Times", "India", None, 42])
def test_is_trusted_source_rejects_generic_or_invalid(A, name):
    assert A.is_trusted_source(name) is False


# ---- legacy-compat sanity on real feature extraction ----
def test_extract_features_empty_text(A):
    feats, up, pu = A.extract_features("")
    assert feats == [0] * 8 and up == 0 and pu == 0


def test_extract_features_counts(A):
    feats, _, _ = A.extract_features("Wow! Really?")
    assert feats[0] == 1 and feats[1] == 1


def test_clickbait_score_in_range_and_higher_for_clickbait(A):
    calm = A.compute_clickbait_score(SAMPLE_TEXT, 0.02, 0.02)
    loud = A.compute_clickbait_score("SHOCKING!!! You won't believe this SECRET", 0.5, 0.1)
    assert 0 <= calm <= 1 and 0 <= loud <= 1 and loud > calm

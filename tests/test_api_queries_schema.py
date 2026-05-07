"""Tests for /api/chat/queries response shape."""


def test_chat_queries_includes_default_for_top_rainy_days(client):
    r = client.get("/api/chat/queries")
    assert r.status_code == 200
    queries = {q["id"]: q for q in r.json()}
    assert "top_rainy_days" in queries
    n_param = queries["top_rainy_days"]["params"]["n"]
    assert n_param["default"] == 10


def test_chat_queries_omits_default_when_param_has_none(client):
    r = client.get("/api/chat/queries")
    queries = {q["id"]: q for q in r.json()}
    # station_id has no default; field should be absent.
    assert "default" not in queries["station_summary"]["params"]["station_id"]

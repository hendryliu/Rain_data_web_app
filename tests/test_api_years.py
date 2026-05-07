"""Tests for the /api/years endpoint."""


def test_api_years_returns_sorted_list(client):
    r = client.get("/api/years")
    assert r.status_code == 200
    years = r.json()
    assert years == [2020]


def test_api_years_with_two_years(fixture_processed_dir_two_years, monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    from app.main import app
    monkeypatch.setattr(main, "PROCESSED_DIR", str(fixture_processed_dir_two_years))
    c = TestClient(app)
    r = c.get("/api/years")
    assert r.status_code == 200
    assert r.json() == [2020, 2021]

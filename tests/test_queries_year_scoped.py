"""Functional tests verifying analytical queries work on the partitioned layout.

The default fixture has S99 with 200 days of data in 2020, every reading 1.0
mm. Total = 200 * 288 = 57,600 mm. Multi-year tests use the two-year fixture.
"""

from app import queries


def test_monthly_totals_returns_only_requested_year(fixture_processed_dir):
    out = queries.monthly_totals("S99", 2020)
    assert "2020" in out["title"]
    # 200 days starting 2020-01-01 → roughly 6.5 months of data; expect 7 month buckets.
    assert 6 <= len(out["data"]["labels"]) <= 7
    # Each reading is 1.0 mm and there are 288 readings/day → values are large positive.
    assert all(v > 0 for v in out["data"]["values"])


def test_monthly_totals_for_year_with_no_file_returns_empty(fixture_processed_dir):
    out = queries.monthly_totals("S99", 2019)
    assert out["data"]["labels"] == []
    assert out["data"]["values"] == []


def test_yearly_totals_lists_all_available_years(fixture_processed_dir_two_years):
    out = queries.yearly_totals("S99")
    # Both fixture years should appear (one or both may be marked partial via "*").
    labels_clean = [lab.rstrip("*") for lab in out["data"]["labels"]]
    assert set(labels_clean) == {"2020", "2021"}


def test_yearly_totals_single_year(fixture_processed_dir):
    out = queries.yearly_totals("S99")
    labels_clean = [lab.rstrip("*") for lab in out["data"]["labels"]]
    assert labels_clean == ["2020"]


def test_hourly_pattern_year_scoped(fixture_processed_dir):
    out = queries.hourly_pattern("S99", year=2020)
    assert "2020" in out["title"]
    assert len(out["data"]["labels"]) == 24  # one bar per hour of day


def test_hourly_pattern_no_year_uses_all_years(fixture_processed_dir):
    out = queries.hourly_pattern("S99")
    # No year in title; still 24 hour buckets.
    assert len(out["data"]["labels"]) == 24


def test_compare_stations_year_scoped(fixture_processed_dir):
    # Comparing the station against itself for one year — both series should match.
    out = queries.compare_stations("S99", "S99", year=2020)
    assert "2020" in out["title"]
    s = out["data"]["series"]
    assert s[0]["values"] == s[1]["values"]


def test_longest_dry_spell_no_dry_days_in_fixture(fixture_processed_dir):
    # Fixture has 1.0 mm every reading → 0 dry days.
    out = queries.longest_dry_spell("S99", year=2020)
    assert "No dry days" in out["text"]


def test_rainiest_week_finds_a_week(fixture_processed_dir):
    out = queries.rainiest_week("S99", year=2020)
    assert "Rainiest 7-day period" in out["text"]


def test_top_rainy_days_returns_n_rows(fixture_processed_dir):
    out = queries.top_rainy_days("S99", year=2020, n=5)
    assert len(out["rows"]) == 5


def test_station_summary_year_scoped(fixture_processed_dir):
    out = queries.station_summary("S99", year=2020)
    assert "2020" in out["title"]
    rows = dict(out["rows"])
    # 200 days × 288 readings/day × 1.0 mm = 57,600 mm total.
    assert "57600" in rows["Total Rainfall"] or "57,600" in rows["Total Rainfall"]

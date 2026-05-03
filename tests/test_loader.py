"""Tests for _load_station's partitioned-layout behavior."""

import pandas as pd
import pytest

from app import queries


class TestLoadStation:
    def test_year_scoped_returns_only_that_year(self, fixture_processed_dir):
        df = queries._load_station("S99", year=2020)
        assert len(df) > 0
        assert (df["timestamp"].dt.year == 2020).all()

    def test_other_year_scoped_returns_only_that_year(self, fixture_processed_dir_two_years):
        df = queries._load_station("S99", year=2021)
        assert len(df) > 0
        assert (df["timestamp"].dt.year == 2021).all()

    def test_missing_year_returns_empty_not_error(self, fixture_processed_dir):
        df = queries._load_station("S99", year=9999)
        assert df.empty
        assert list(df.columns) == ["timestamp", "reading_value"]

    def test_unknown_station_raises(self, fixture_processed_dir):
        with pytest.raises(ValueError, match="No data for station UNKNOWN"):
            queries._load_station("UNKNOWN")

    def test_no_year_returns_all_years_concatenated(self, fixture_processed_dir_two_years):
        all_df = queries._load_station("S99")
        df_2020 = queries._load_station("S99", year=2020)
        df_2021 = queries._load_station("S99", year=2021)
        assert len(all_df) == len(df_2020) + len(df_2021)
        years = set(all_df["timestamp"].dt.year.unique())
        assert years == {2020, 2021}

    def test_loader_strips_tz(self, fixture_processed_dir):
        df = queries._load_station("S99", year=2020)
        assert df["timestamp"].dt.tz is None

    def test_loader_casts_to_float64(self, fixture_processed_dir):
        df = queries._load_station("S99", year=2020)
        assert df["reading_value"].dtype == "float64"


class TestAvailableYears:
    def test_single_year_fixture(self, fixture_processed_dir):
        assert queries._available_years("S99") == [2020]

    def test_two_year_fixture_returns_sorted(self, fixture_processed_dir_two_years):
        assert queries._available_years("S99") == [2020, 2021]

    def test_unknown_station_raises(self, fixture_processed_dir):
        with pytest.raises(ValueError, match="No data for station UNKNOWN"):
            queries._available_years("UNKNOWN")

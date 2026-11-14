from pathlib import Path

from quantlab.data import validate_matches_csv


DATASET = Path(__file__).resolve().parents[1] / "data" / "epl_pilot" / "epl_results_2010_2026.csv"


def test_epl_pilot_dataset_passes_production_csv_validation():
    matches, errors = validate_matches_csv(DATASET)
    assert errors == []
    assert len(matches) == 6130
    assert matches["match_id"].is_unique
    assert set(matches["league"]) == {"EPL"}
    assert matches["home_goals"].ge(0).all()
    assert matches["away_goals"].ge(0).all()
    assert matches["date"].notna().all()
    assert "home_xg" not in matches.columns
    assert "away_xg" not in matches.columns


def test_epl_pilot_season_coverage_and_current_season_is_partial():
    matches, errors = validate_matches_csv(DATASET)
    assert errors == []
    counts = matches.groupby("season").size().to_dict()
    completed = [f"{year}-{str(year + 1)[-2:]}" for year in range(2010, 2026)]
    assert all(counts[season] == 380 for season in completed)
    assert counts["2026-27"] == 50

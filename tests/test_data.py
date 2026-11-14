import io
from quantlab.data import validate_matches_csv


def test_csv_rejects_fractional_goals():
    data = "date,home_team,away_team,home_goals,away_goals\n2026-01-01,A,B,1.5,0\n"
    _, errors = validate_matches_csv(io.StringIO(data))
    assert any("whole-number" in e for e in errors)


def test_csv_generates_match_id():
    data = "date,home_team,away_team,home_goals,away_goals\n2026-01-01,A,B,1,0\n"
    df, errors = validate_matches_csv(io.StringIO(data))
    assert not errors
    assert "match_id" in df.columns

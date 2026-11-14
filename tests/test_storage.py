from datetime import datetime, timedelta, timezone
import sqlite3
import pytest
from quantlab.storage import save_prediction, settle_prediction, list_predictions


def test_prediction_requires_future_timezone_aware_kickoff(tmp_path):
    db = tmp_path / "test.sqlite3"
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    with pytest.raises(ValueError):
        save_prediction("m1", past, "A", "B", {"home_win": 0.5}, db_path=db)
    future_naive = (datetime.now() + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError):
        save_prediction("m2", future_naive, "A", "B", {"home_win": 0.5}, db_path=db)


def test_prediction_cannot_be_settled_twice(tmp_path):
    db = tmp_path / "test.sqlite3"
    kickoff = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    pid = save_prediction("m1", kickoff, "A", "B", {"home_win": 0.5}, db_path=db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE predictions SET kickoff=? WHERE id=?",
                     ((datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat(), pid))
    settle_prediction(pid, 2, 1, 5.0, db)
    with pytest.raises(ValueError):
        settle_prediction(pid, 0, 0, -1.0, db)
    row = list_predictions(db).iloc[0]
    assert row.actual_home_goals == 2
    assert row.simulated_profit == 5.0

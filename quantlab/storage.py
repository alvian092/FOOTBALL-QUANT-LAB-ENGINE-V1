"""SQLite persistence for datasets and immutable prediction records."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

DEFAULT_DB = Path("data/football_quant_lab.sqlite3")


def connect(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize(db_path: str | Path = DEFAULT_DB) -> None:
    with connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
          id INTEGER PRIMARY KEY AUTOINCREMENT, uploaded_at TEXT NOT NULL,
          filename TEXT NOT NULL, row_count INTEGER NOT NULL, content_hash TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS matches (
          match_id TEXT PRIMARY KEY, date TEXT NOT NULL, home_team TEXT NOT NULL,
          away_team TEXT NOT NULL, home_goals INTEGER NOT NULL, away_goals INTEGER NOT NULL,
          dataset_id INTEGER, FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        );
        CREATE TABLE IF NOT EXISTS predictions (
          id INTEGER PRIMARY KEY AUTOINCREMENT, match_id TEXT NOT NULL, kickoff TEXT NOT NULL,
          home_team TEXT NOT NULL, away_team TEXT NOT NULL, probabilities_json TEXT NOT NULL,
          selected_market TEXT, odds REAL, ev REAL, model_version TEXT NOT NULL,
          predicted_at TEXT NOT NULL, actual_home_goals INTEGER, actual_away_goals INTEGER,
          simulated_profit REAL, UNIQUE(match_id, model_version, predicted_at)
        );
        """)
        conn.commit()


def save_dataset(df: pd.DataFrame, filename: str, content_hash: str,
                 db_path: str | Path = DEFAULT_DB) -> int:
    initialize(db_path)
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO datasets(uploaded_at,filename,row_count,content_hash) VALUES(?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), filename, len(df), content_hash))
        row = conn.execute("SELECT id FROM datasets WHERE content_hash=?", (content_hash,)).fetchone()
        dataset_id = int(row[0])
        for r in df.itertuples():
            conn.execute("""INSERT OR IGNORE INTO matches
                (match_id,date,home_team,away_team,home_goals,away_goals,dataset_id)
                VALUES(?,?,?,?,?,?,?)""",
                (str(r.match_id), r.date.isoformat(), str(r.home_team), str(r.away_team),
                 int(r.home_goals), int(r.away_goals), dataset_id))
        conn.commit()
        return dataset_id


def load_matches(db_path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    initialize(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM matches ORDER BY date", conn)


def save_prediction(match_id: str, kickoff: str, home_team: str, away_team: str,
                    probabilities: dict, selected_market: str | None = None,
                    odds: float | None = None, ev: float | None = None,
                    model_version: str = "1.0.0", db_path: str | Path = DEFAULT_DB) -> int:
    initialize(db_path)
    with connect(db_path) as conn:
        cur = conn.execute("""INSERT INTO predictions
          (match_id,kickoff,home_team,away_team,probabilities_json,selected_market,odds,ev,model_version,predicted_at)
          VALUES(?,?,?,?,?,?,?,?,?,?)""",
          (match_id,kickoff,home_team,away_team,json.dumps(probabilities),selected_market,odds,ev,
           model_version,datetime.now(timezone.utc).isoformat()))
        conn.commit()
        return int(cur.lastrowid)


def settle_prediction(prediction_id: int, home_goals: int, away_goals: int,
                      profit: float, db_path: str | Path = DEFAULT_DB) -> None:
    initialize(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT kickoff FROM predictions WHERE id=?", (prediction_id,)).fetchone()
        if row is None:
            raise ValueError("Prediction not found.")
        if pd.Timestamp.now(tz="UTC") < pd.Timestamp(row[0]):
            raise ValueError("Cannot settle a prediction before its kickoff.")
        conn.execute("""UPDATE predictions SET actual_home_goals=?,actual_away_goals=?,simulated_profit=?
                        WHERE id=? AND actual_home_goals IS NULL""",
                     (home_goals,away_goals,profit,prediction_id))
        conn.commit()


def list_predictions(db_path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    initialize(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY predicted_at DESC", conn)

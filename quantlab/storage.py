"""SQLite persistence, immutable prediction snapshots, and safe backup/restore."""
from __future__ import annotations
import json
import math
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

DEFAULT_DB = Path("data/football_quant_lab.sqlite3")


def connect(db_path=DEFAULT_DB):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize(db_path=DEFAULT_DB):
    with connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
          id INTEGER PRIMARY KEY AUTOINCREMENT, uploaded_at TEXT NOT NULL,
          filename TEXT NOT NULL, row_count INTEGER NOT NULL, content_hash TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS matches (
          match_id TEXT PRIMARY KEY, date TEXT NOT NULL, home_team TEXT NOT NULL,
          away_team TEXT NOT NULL, home_goals INTEGER NOT NULL, away_goals INTEGER NOT NULL,
          dataset_id INTEGER, home_xg REAL, away_xg REAL, league TEXT,
          FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        );
        CREATE TABLE IF NOT EXISTS odds_snapshots (
          id INTEGER PRIMARY KEY AUTOINCREMENT, match_id TEXT NOT NULL, market TEXT NOT NULL,
          selection TEXT NOT NULL, bookmaker TEXT NOT NULL DEFAULT 'Unknown', odds REAL NOT NULL,
          captured_at TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'manual',
          UNIQUE(match_id, market, selection, bookmaker, captured_at)
        );
        CREATE TABLE IF NOT EXISTS predictions (
          id INTEGER PRIMARY KEY AUTOINCREMENT, match_id TEXT NOT NULL, kickoff TEXT NOT NULL,
          home_team TEXT NOT NULL, away_team TEXT NOT NULL, probabilities_json TEXT NOT NULL,
          selected_market TEXT, odds REAL, ev REAL, model_version TEXT NOT NULL,
          predicted_at TEXT NOT NULL, actual_home_goals INTEGER, actual_away_goals INTEGER,
          simulated_profit REAL, stake REAL NOT NULL DEFAULT 0, closing_odds REAL,
          UNIQUE(match_id, model_version, predicted_at)
        );
        """)
        mcols = {r[1] for r in conn.execute("PRAGMA table_info(matches)")}
        for name, declaration in (("home_xg","REAL"),("away_xg","REAL"),("league","TEXT")):
            if name not in mcols: conn.execute(f"ALTER TABLE matches ADD COLUMN {name} {declaration}")
        pcols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
        for name, declaration in (("stake", "REAL NOT NULL DEFAULT 0"), ("closing_odds", "REAL")):
            if name not in pcols: conn.execute(f"ALTER TABLE predictions ADD COLUMN {name} {declaration}")
        conn.commit()


def save_dataset(df, filename, content_hash, db_path=DEFAULT_DB):
    initialize(db_path)
    has_xg = {"home_xg", "away_xg"}.issubset(df.columns)
    with connect(db_path) as conn:
        conn.execute("INSERT OR IGNORE INTO datasets(uploaded_at,filename,row_count,content_hash) VALUES(?,?,?,?)",
                     (datetime.now(timezone.utc).isoformat(), filename, len(df), content_hash))
        dataset_id = int(conn.execute("SELECT id FROM datasets WHERE content_hash=?", (content_hash,)).fetchone()[0])
        for r in df.itertuples():
            hxg, axg = (float(r.home_xg), float(r.away_xg)) if has_xg else (None, None)
            league = str(r.league) if "league" in df.columns else None
            conn.execute("""INSERT OR IGNORE INTO matches
                (match_id,date,home_team,away_team,home_goals,away_goals,dataset_id,home_xg,away_xg,league)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (str(r.match_id), r.date.isoformat(), str(r.home_team), str(r.away_team),
                 int(r.home_goals), int(r.away_goals), dataset_id, hxg, axg, league))
        conn.commit()
        return dataset_id


def load_matches(db_path=DEFAULT_DB):
    initialize(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM matches ORDER BY date", conn)


def save_prediction(match_id, kickoff, home_team, away_team, probabilities,
                    selected_market=None, odds=None, ev=None, model_version="1.0.0",
                    stake=0.0, db_path=DEFAULT_DB):
    initialize(db_path)
    kickoff_ts = pd.Timestamp(kickoff)
    if kickoff_ts.tzinfo is None: raise ValueError("Kickoff must include a timezone.")
    if kickoff_ts.tz_convert("UTC") <= pd.Timestamp.now(tz="UTC"): raise ValueError("Prediction must be saved before kickoff.")
    if not match_id.strip() or not home_team.strip() or not away_team.strip() or home_team == away_team:
        raise ValueError("Valid match ID and two different team names are required.")
    if stake < 0: raise ValueError("Stake cannot be negative.")
    with connect(db_path) as conn:
        cur = conn.execute("""INSERT INTO predictions
          (match_id,kickoff,home_team,away_team,probabilities_json,selected_market,odds,ev,model_version,predicted_at,stake)
          VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
          (match_id,kickoff_ts.isoformat(),home_team,away_team,json.dumps(probabilities),
           selected_market,odds,ev,model_version,datetime.now(timezone.utc).isoformat(),stake))
        conn.commit()
        return int(cur.lastrowid)


def record_closing_odds(prediction_id, closing_odds, db_path=DEFAULT_DB):
    initialize(db_path)
    if closing_odds is None or float(closing_odds) <= 1: raise ValueError("Closing odds must be greater than 1.")
    with connect(db_path) as conn:
        row = conn.execute("SELECT kickoff,selected_market FROM predictions WHERE id=?", (prediction_id,)).fetchone()
        if row is None: raise ValueError("Prediction not found.")
        if not row[1]: raise ValueError("NO BET records do not have closing odds.")
        if pd.Timestamp.now(tz="UTC") >= pd.Timestamp(row[0]).tz_convert("UTC"):
            raise ValueError("Closing odds must be recorded before kickoff.")
        conn.execute("UPDATE predictions SET closing_odds=? WHERE id=?", (float(closing_odds), prediction_id))
        conn.commit()


def settle_prediction(prediction_id, home_goals, away_goals, profit, db_path=DEFAULT_DB):
    initialize(db_path)
    if isinstance(home_goals, bool) or isinstance(away_goals, bool) or home_goals < 0 or away_goals < 0:
        raise ValueError("Actual goals must be non-negative integers.")
    if int(home_goals) != home_goals or int(away_goals) != away_goals: raise ValueError("Actual goals must be integers.")
    with connect(db_path) as conn:
        row = conn.execute("SELECT kickoff,actual_home_goals FROM predictions WHERE id=?", (prediction_id,)).fetchone()
        if row is None: raise ValueError("Prediction not found.")
        if pd.Timestamp.now(tz="UTC") < pd.Timestamp(row[0]).tz_convert("UTC"): raise ValueError("Cannot settle before kickoff.")
        if row[1] is not None: raise ValueError("Prediction has already been settled and is immutable.")
        conn.execute("""UPDATE predictions SET actual_home_goals=?,actual_away_goals=?,simulated_profit=?
                        WHERE id=? AND actual_home_goals IS NULL""",
                     (int(home_goals),int(away_goals),float(profit),prediction_id))
        conn.commit()


def list_predictions(db_path=DEFAULT_DB):
    initialize(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY predicted_at DESC", conn)


def list_datasets(db_path=DEFAULT_DB):
    initialize(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM datasets ORDER BY uploaded_at DESC", conn)


def backup_database(db_path=DEFAULT_DB, backup_path="football_quant_lab_backup.sqlite3"):
    """Create a transactionally consistent SQLite snapshot."""
    source = Path(db_path)
    if not source.exists(): raise FileNotFoundError(source)
    target = Path(backup_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
        src.backup(dst)
    return target


def restore_database(backup_path, db_path=DEFAULT_DB):
    """Validate a backup before atomically replacing the active database."""
    source, target = Path(backup_path), Path(db_path)
    if not source.exists(): raise FileNotFoundError(source)
    with sqlite3.connect(source) as conn:
        check = conn.execute("PRAGMA integrity_check").fetchone()
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not check or check[0] != "ok" or not {"datasets","matches","predictions"}.issubset(tables):
            raise ValueError("Backup is invalid or missing required Football Quant Lab tables.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".restore.tmp")
    try:
        with sqlite3.connect(source) as src, sqlite3.connect(temp) as dst: src.backup(dst)
        temp.replace(target)
    finally:
        if temp.exists(): temp.unlink()
    initialize(target)


def save_odds_snapshot(match_id, market, odds_by_selection, captured_at, bookmaker="Unknown", source="manual", db_path=DEFAULT_DB):
    """Persist one immutable, timestamped market snapshot with all selections."""
    initialize(db_path)
    stamp = pd.Timestamp(captured_at)
    if stamp.tzinfo is None: raise ValueError("captured_at must include a timezone.")
    if not str(match_id).strip() or not str(market).strip() or not isinstance(odds_by_selection, dict) or len(odds_by_selection) < 2:
        raise ValueError("Provide match_id, market, and at least two mutually exclusive selections.")
    if any(not str(k).strip() or not math.isfinite(float(v)) or float(v) <= 1
           for k,v in odds_by_selection.items()):
        raise ValueError("Selections must be named and all decimal odds must be finite and greater than 1.")
    with connect(db_path) as conn:
        conn.executemany("""INSERT OR IGNORE INTO odds_snapshots
            (match_id,market,selection,bookmaker,odds,captured_at,source) VALUES(?,?,?,?,?,?,?)""",
            [(str(match_id),str(market),str(k),str(bookmaker),float(v),stamp.isoformat(),str(source))
             for k,v in odds_by_selection.items()])
        conn.commit()


def list_odds_snapshots(db_path=DEFAULT_DB):
    initialize(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM odds_snapshots ORDER BY captured_at", conn)

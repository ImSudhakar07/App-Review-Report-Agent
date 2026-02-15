"""
Database layer — the storage backbone of the App Store Agent.

Uses SQLite: a file-based database built into Python.
Each app gets its own .db file for complete data isolation.

Key concept — SQL (Structured Query Language):
    SQL is the language used to talk to databases. You'll see commands like:
    - CREATE TABLE: defines a new table
    - INSERT INTO: adds a row
    - SELECT: reads data
    - WHERE: filters rows
    Think of it as writing very precise questions to a spreadsheet.
"""

import sqlite3
import os
import json
from datetime import datetime, date
from typing import Optional
from app.models import Review, AppInfo
from app.config import DATABASE_DIR


def list_analyzed_apps() -> list[dict]:
    """
    Scan the database directory and return info about all previously analyzed apps.
    Each app has its own .db file — we read metadata from each one.
    """
    apps = []
    if not os.path.exists(DATABASE_DIR):
        return apps

    for filename in os.listdir(DATABASE_DIR):
        if filename.endswith(".db"):
            db_path = os.path.join(DATABASE_DIR, filename)
            try:
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # Check if app_metadata table exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='app_metadata'"
                )
                if cursor.fetchone():
                    cursor.execute("SELECT key, value FROM app_metadata")
                    meta = {row["key"]: row["value"] for row in cursor.fetchall()}
                    if meta.get("app_id"):
                        apps.append(meta)
                conn.close()
            except Exception:
                continue

    return apps


def delete_app_data(app_id: str) -> int:
    """
    Delete ALL data for an app — reviews, analysis, themes, metadata.
    Returns the number of reviews that were deleted.
    """
    db_path = _get_db_path(app_id)
    if not os.path.exists(db_path):
        return 0

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    review_count = 0
    try:
        cursor.execute("SELECT COUNT(*) FROM reviews")
        review_count = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        pass
    conn.close()

    # Delete the entire database file
    os.remove(db_path)
    return review_count


def get_analyzed_months(app_id: str) -> list[str]:
    """
    Return list of period_labels that have already been analyzed (monthly).
    Used to skip re-analysis of already-processed months.
    """
    conn = _get_connection(app_id)
    cursor = conn.cursor()
    if not _table_exists(cursor, "period_analysis"):
        conn.close()
        return []
    cursor.execute(
        "SELECT period_label FROM period_analysis WHERE period_type = 'monthly' ORDER BY period_start ASC"
    )
    result = [row[0] for row in cursor.fetchall()]
    conn.close()
    return result


def count_unanalyzed_reviews(app_id: str, start_date: str, end_date: str) -> tuple[int, int]:
    """
    Count reviews in the period that belong to months NOT yet analyzed.
    Returns (unanalyzed_count, total_in_period).
    """
    from app.processor import get_month_ranges

    total = count_reviews_for_period(app_id, start_date, end_date)
    already_done = set(get_analyzed_months(app_id))
    months = get_month_ranges(start_date, end_date)

    unanalyzed = 0
    for label, m_start, m_end in months:
        if label not in already_done:
            unanalyzed += count_reviews_for_period(app_id, m_start, m_end)

    return unanalyzed, total


def get_last_scraped_date(app_id: str) -> str:
    """
    Get the date of the most recent review in the database.
    Used to tell the user 'you have data up to this date'.
    """
    conn = _get_connection(app_id)
    cursor = conn.cursor()
    if not _table_exists(cursor, "reviews"):
        conn.close()
        return ""
    cursor.execute("SELECT MAX(date) FROM reviews")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row and row[0] else ""


def delete_analysis_only(app_id: str) -> None:
    """
    Delete analysis results but keep raw reviews.
    Used for re-running analysis with different LLM settings.
    """
    conn = _get_connection(app_id)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM period_analysis")
        cursor.execute("DELETE FROM themes")
        cursor.execute("UPDATE app_metadata SET value = '' WHERE key = 'last_analyzed_date'")
        cursor.execute("UPDATE app_metadata SET value = 'false' WHERE key = 'seagull_analysis_complete'")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.close()


def aggregate_themes_from_monthly(app_id: str, period_type: str, period_label: str,
                                   monthly_labels: list[str]) -> list[dict]:
    """
    Aggregate themes from multiple monthly analyses into a quarterly/yearly summary.
    Instead of re-running LLM on all reviews, we combine monthly theme counts.
    This preserves the statistical significance from monthly analysis.
    """
    conn = _get_connection(app_id)
    cursor = conn.cursor()
    if not _table_exists(cursor, "themes"):
        conn.close()
        return []

    # Collect all monthly themes
    theme_map = {}  # {(theme_name, sentiment): {total_count, samples, confidences}}

    for m_label in monthly_labels:
        cursor.execute(
            "SELECT * FROM themes WHERE period_type = 'monthly' AND period_label = ?",
            (m_label,)
        )
        for row in cursor.fetchall():
            row = dict(row)
            key = (row["theme"], row["sentiment"])
            if key not in theme_map:
                theme_map[key] = {
                    "theme": row["theme"],
                    "sentiment": row["sentiment"],
                    "mention_count": 0,
                    "sample_reviews": [],
                    "confidences": [],
                }
            theme_map[key]["mention_count"] += row.get("mention_count", 0)
            try:
                samples = json.loads(row.get("sample_reviews", "[]"))
                theme_map[key]["sample_reviews"].extend(samples)
            except (TypeError, json.JSONDecodeError):
                pass
            theme_map[key]["confidences"].append(row.get("confidence", 0))

    conn.close()

    # Build aggregated theme list
    result = []
    for key, data in theme_map.items():
        avg_conf = sum(data["confidences"]) / len(data["confidences"]) if data["confidences"] else 0
        result.append({
            "theme": data["theme"],
            "sentiment": data["sentiment"],
            "mention_count": data["mention_count"],
            "sample_reviews": json.dumps(data["sample_reviews"][:5]),  # Keep top 5 samples
            "confidence": round(avg_conf, 2),
        })

    # Sort by mention count
    result.sort(key=lambda x: x["mention_count"], reverse=True)
    return result


def _get_db_path(app_id: str) -> str:
    """
    Each app gets its own database file.
    e.g., "com.spotify.music" -> "data/processed/com_spotify_music.db"

    This is the multi-app isolation you specified.
    Spotify data never mixes with Instagram data.
    """
    # Replace dots with underscores for a clean filename
    safe_name = app_id.replace(".", "_").replace(" ", "_")
    os.makedirs(DATABASE_DIR, exist_ok=True)
    return os.path.join(DATABASE_DIR, f"{safe_name}.db")


def _get_connection(app_id: str) -> sqlite3.Connection:
    """
    Open a connection to the app's database.

    What's a "connection"?
    Think of it like opening a file. You open it, read/write, then close it.
    A database connection is the same — it's your open channel to the database.
    """
    db_path = _get_db_path(app_id)
    conn = sqlite3.connect(db_path)

    # This makes SQLite return rows as dictionaries instead of plain tuples.
    # Without it: row = (1, "great app", 5)
    # With it:    row = {"id": 1, "text": "great app", "rating": 5}
    conn.row_factory = sqlite3.Row

    return conn


def initialize_database(app_id: str, app_name: str, store: str) -> None:
    """
    Creates all tables for a new app. Safe to call multiple times —
    'IF NOT EXISTS' means it won't crash if the tables already exist.

    This runs once when you first analyze an app.
    """
    conn = _get_connection(app_id)
    cursor = conn.cursor()

    # ---- Table 1: reviews ----
    # Stores every individual review
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id   TEXT PRIMARY KEY,
            source      TEXT NOT NULL,
            rating      INTEGER NOT NULL,
            text        TEXT,
            date        TEXT NOT NULL,
            username    TEXT,
            thumbs_up   INTEGER DEFAULT 0,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ---- Table 2: period_analysis ----
    # Aggregated stats for each time period (week, month, quarter, year)
    # This is what Agent 2 reads to benchmark against
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS period_analysis (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            period_type     TEXT NOT NULL,
            period_label    TEXT NOT NULL,
            period_start    TEXT NOT NULL,
            period_end      TEXT NOT NULL,
            total_reviews   INTEGER DEFAULT 0,
            rating_1        INTEGER DEFAULT 0,
            rating_2        INTEGER DEFAULT 0,
            rating_3        INTEGER DEFAULT 0,
            rating_4        INTEGER DEFAULT 0,
            rating_5        INTEGER DEFAULT 0,
            avg_rating      REAL DEFAULT 0.0,
            reviews_with_text   INTEGER DEFAULT 0,
            reviews_without_text INTEGER DEFAULT 0,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(period_type, period_label)
        )
    """)

    # ---- Table 3: themes ----
    # Top positive and negative themes per period
    # Each row is one theme for one period
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS themes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            period_type     TEXT NOT NULL,
            period_label    TEXT NOT NULL,
            theme           TEXT NOT NULL,
            sentiment       TEXT NOT NULL,
            mention_count   INTEGER DEFAULT 0,
            sample_reviews  TEXT,
            confidence      REAL DEFAULT 0.0,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ---- Table 4: app_metadata ----
    # Tracks the app's analysis state
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS app_metadata (
            key     TEXT PRIMARY KEY,
            value   TEXT NOT NULL
        )
    """)

    # Insert initial metadata
    metadata_defaults = {
        "app_id": app_id,
        "app_name": app_name,
        "store": store,
        "last_analyzed_date": "",
        "last_analyzed_week_ending": "",
        "total_reviews_stored": "0",
        "seagull_analysis_complete": "false",
    }
    for key, value in metadata_defaults.items():
        cursor.execute(
            "INSERT OR IGNORE INTO app_metadata (key, value) VALUES (?, ?)",
            (key, value)
        )

    conn.commit()  # Save all changes to disk
    conn.close()   # Close the connection

    print(f"Database initialized for: {app_name} ({app_id})")
    print(f"Database file: {_get_db_path(app_id)}")


def store_reviews(app_id: str, reviews: list[Review]) -> int:
    """
    Save reviews to the database.
    Uses INSERT OR IGNORE — if a review with the same ID already exists,
    it's skipped (no duplicates). This is key for incremental processing:
    when Agent 2 runs weekly, it won't re-insert old reviews.

    Returns the number of new reviews actually inserted.
    """
    conn = _get_connection(app_id)
    cursor = conn.cursor()

    inserted = 0
    for review in reviews:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO reviews
                (review_id, source, rating, text, date, username, thumbs_up)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                review.review_id,
                review.source,
                review.rating,
                review.text,
                review.date.isoformat(),   # Store dates as ISO strings (e.g., "2026-01-15T10:30:00")
                review.username,
                review.thumbs_up
            ))
            if cursor.rowcount > 0:
                inserted += 1
        except sqlite3.Error as e:
            print(f"  Error storing review {review.review_id}: {e}")

    # Update total count in metadata
    cursor.execute("SELECT COUNT(*) FROM reviews")
    total = cursor.fetchone()[0]
    cursor.execute(
        "UPDATE app_metadata SET value = ? WHERE key = 'total_reviews_stored'",
        (str(total),)
    )

    conn.commit()
    conn.close()

    print(f"Stored {inserted} new reviews ({len(reviews) - inserted} duplicates skipped)")
    return inserted


def count_reviews_for_period(app_id: str, start_date: str, end_date: str) -> int:
    """
    Count how many reviews exist in the database for a given date range.
    Fast query — doesn't load the actual data, just counts.
    """
    conn = _get_connection(app_id)
    cursor = conn.cursor()
    if not _table_exists(cursor, "reviews"):
        conn.close()
        return 0
    cursor.execute(
        "SELECT COUNT(*) FROM reviews WHERE date >= ? AND date <= ?",
        (start_date, end_date)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_review_date_range(app_id: str) -> tuple:
    """Get the earliest and latest review dates for an app."""
    conn = _get_connection(app_id)
    cursor = conn.cursor()
    if not _table_exists(cursor, "reviews"):
        conn.close()
        return None, None
    cursor.execute("SELECT MIN(date), MAX(date) FROM reviews")
    row = cursor.fetchone()
    conn.close()
    return row[0], row[1]


def get_reviews_for_period(app_id: str, start_date: str, end_date: str) -> list[dict]:
    """
    Retrieve reviews within a date range.
    This is what both agents use to pull reviews for analysis.

    Args:
        start_date: ISO format date string (e.g., "2025-06-01")
        end_date:   ISO format date string (e.g., "2025-06-30")

    Returns:
        List of review dictionaries.
    """
    conn = _get_connection(app_id)
    cursor = conn.cursor()

    if not _table_exists(cursor, "reviews"):
        conn.close()
        return []

    cursor.execute("""
        SELECT * FROM reviews
        WHERE date >= ? AND date <= ?
        ORDER BY date ASC
    """, (start_date, end_date))

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def store_period_analysis(app_id: str, analysis: dict) -> None:
    """
    Save aggregated analysis for a period.
    Uses INSERT OR REPLACE — if analysis for this period already exists,
    it gets updated (not duplicated).
    """
    conn = _get_connection(app_id)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO period_analysis
        (period_type, period_label, period_start, period_end,
         total_reviews, rating_1, rating_2, rating_3, rating_4, rating_5,
         avg_rating, reviews_with_text, reviews_without_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        analysis["period_type"],
        analysis["period_label"],
        analysis["period_start"],
        analysis["period_end"],
        analysis["total_reviews"],
        analysis["rating_1"],
        analysis["rating_2"],
        analysis["rating_3"],
        analysis["rating_4"],
        analysis["rating_5"],
        analysis["avg_rating"],
        analysis["reviews_with_text"],
        analysis["reviews_without_text"],
    ))

    conn.commit()
    conn.close()


def store_themes(app_id: str, period_type: str, period_label: str, themes: list[dict]) -> None:
    """
    Save extracted themes for a period.
    Clears old themes for that period first, then inserts fresh ones.
    """
    conn = _get_connection(app_id)
    cursor = conn.cursor()

    # Remove old themes for this period (so re-analysis overwrites cleanly)
    cursor.execute(
        "DELETE FROM themes WHERE period_type = ? AND period_label = ?",
        (period_type, period_label)
    )

    for theme in themes:
        cursor.execute("""
            INSERT INTO themes
            (period_type, period_label, theme, sentiment, mention_count,
             sample_reviews, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            period_type,
            period_label,
            theme["theme"],
            theme["sentiment"],
            theme["mention_count"],
            # Convert list to JSON string — SQLite can only store simple types
            json.dumps(theme.get("sample_reviews", [])) if isinstance(theme.get("sample_reviews"), list) else str(theme.get("sample_reviews", "")),
            theme.get("confidence", 0.0),
        ))

    conn.commit()
    conn.close()


def update_metadata(app_id: str, key: str, value: str) -> None:
    """Update a single metadata value."""
    conn = _get_connection(app_id)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE app_metadata SET value = ? WHERE key = ?",
        (value, key)
    )
    conn.commit()
    conn.close()


def get_metadata(app_id: str) -> dict:
    """Get all metadata for an app as a dictionary."""
    conn = _get_connection(app_id)
    cursor = conn.cursor()
    if not _table_exists(cursor, "app_metadata"):
        conn.close()
        return {}
    cursor.execute("SELECT key, value FROM app_metadata")
    result = {row["key"]: row["value"] for row in cursor.fetchall()}
    conn.close()
    return result


def _table_exists(cursor, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


def get_all_period_analyses(app_id: str, period_type: Optional[str] = None) -> list[dict]:
    """
    Get all period analyses, optionally filtered by type.
    This is what the dashboard and Agent 2 use to show trends.
    Returns empty list if tables don't exist yet (first run).
    """
    conn = _get_connection(app_id)
    cursor = conn.cursor()

    if not _table_exists(cursor, "period_analysis"):
        conn.close()
        return []

    if period_type:
        cursor.execute(
            "SELECT * FROM period_analysis WHERE period_type = ? ORDER BY period_start ASC",
            (period_type,)
        )
    else:
        cursor.execute("SELECT * FROM period_analysis ORDER BY period_start ASC")

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_themes_for_period(app_id: str, period_type: str, period_label: str) -> list[dict]:
    """Get themes for a specific period."""
    conn = _get_connection(app_id)
    cursor = conn.cursor()
    if not _table_exists(cursor, "themes"):
        conn.close()
        return []
    cursor.execute(
        "SELECT * FROM themes WHERE period_type = ? AND period_label = ? ORDER BY mention_count DESC",
        (period_type, period_label)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

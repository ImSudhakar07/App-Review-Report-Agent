"""
Seagull Processor — the brain of Agent 1.

Takes raw reviews from the database, processes them month-by-month,
and produces structured analysis (ratings, themes, anomalies) per period.

Key design decisions:
    1. Process month-by-month to avoid context rot (user requirement).
    2. Split reviews into ratings-only (stats) and text reviews (AI analysis).
    3. Statistical significance filtering for themes.
    4. LLM only handles text analysis — pure stats done with code (cheaper, faster).
"""

import json
from datetime import datetime, timedelta
from collections import Counter
import math

from app.llm_client import call_llm
from app.database import (
    get_reviews_for_period,
    store_period_analysis,
    store_themes,
    update_metadata,
    get_metadata,
    get_analyzed_months,
    aggregate_themes_from_monthly,
)

# ============================================================
# PART 1: Pure statistics (no LLM needed)
# ============================================================

def compute_rating_stats(reviews: list[dict]) -> dict:
    """
    Calculate rating distribution from a list of reviews.
    This is pure math — no AI involved. Fast and free.

    Why separate this from LLM analysis?
    Because you said: "number of reviews categorised by 1 star to 5 star."
    That's a counting problem, not an intelligence problem.
    Never use an LLM for what a calculator can do.
    """
    total = len(reviews)
    if total == 0:
        return {
            "total_reviews": 0, "avg_rating": 0,
            "rating_1": 0, "rating_2": 0, "rating_3": 0,
            "rating_4": 0, "rating_5": 0,
            "reviews_with_text": 0, "reviews_without_text": 0,
        }

    ratings = [r["rating"] for r in reviews]
    rating_counts = Counter(ratings)

    with_text = sum(1 for r in reviews if r.get("text") and len(r["text"].strip()) > 0)

    return {
        "total_reviews": total,
        "avg_rating": round(sum(ratings) / total, 2),
        "rating_1": rating_counts.get(1, 0),
        "rating_2": rating_counts.get(2, 0),
        "rating_3": rating_counts.get(3, 0),
        "rating_4": rating_counts.get(4, 0),
        "rating_5": rating_counts.get(5, 0),
        "reviews_with_text": with_text,
        "reviews_without_text": total - with_text,
    }


def check_statistical_significance(count: int, total: int, min_sample: int = 5) -> bool:
    """
    Simple significance check: does this theme have enough mentions
    to be worth reporting?

    Args:
        count: How many reviews mention this theme.
        total: Total reviews in the period.
        min_sample: Minimum number of mentions required.

    This is a practical filter, not a full hypothesis test.
    A theme mentioned 2 times out of 5000 reviews is noise.
    A theme mentioned 50 times out of 5000 is a signal.

    We'll use a more rigorous z-test for trend comparisons later.
    """
    if count < min_sample:
        return False
    # Theme must appear in at least 1% of reviews OR at least min_sample times
    proportion = count / total if total > 0 else 0
    return proportion >= 0.01 or count >= min_sample


# ============================================================
# PART 2: LLM-powered analysis (the expensive, intelligent part)
# ============================================================

# This is the SYSTEM PROMPT — the "job description" for the LLM.
# Notice how precisely it's written. Every sentence constrains the model.
# This is context engineering in action.

THEME_EXTRACTION_SYSTEM_PROMPT = """You are a rigorous app review analyst. Your job is to extract themes (topics) from user reviews.

RULES — follow these exactly:
1. A "theme" is a specific topic users talk about (e.g., "battery drain", "login issues", "music recommendations").
2. Classify each theme as "positive" or "negative" based on how users feel about it.
3. Count how many reviews mention each theme.
4. For each theme, include 2-3 direct quotes from actual reviews as evidence.
5. Do NOT invent themes. Only report what users actually say.
6. Do NOT paraphrase beyond recognition. Use the customer's own words when possible.
7. If a theme has fewer than the minimum sample size, exclude it.
8. Themes should be specific and actionable, not vague (e.g., "slow loading on startup" not just "performance").

CRITICAL NAMING RULES — follow strictly:
9. Theme names MUST be lowercase, 2-4 words maximum (e.g., "app crashing", "slow loading", "great content").
10. NEVER use filler words like "issues with", "problems with", "quality of". Go straight to the noun/verb.
11. NEVER add qualifiers like "or freezing", "or not opening". Pick ONE canonical short name.
12. If two themes overlap (e.g., "app crashing" and "app freezing"), MERGE them into one theme.
13. Use the SIMPLEST, most GENERAL label. Examples:
    - GOOD: "app crashing" | BAD: "app crashing or freezing", "app crashing or not opening"
    - GOOD: "subscription cost" | BAD: "subscription requirements", "subscription problems", "subscription issues"
    - GOOD: "quality content" | BAD: "high-quality news content", "informative news content", "quality of content"
    - GOOD: "too many ads" | BAD: "excessive advertisements", "too many ads appearing"
14. The same concept MUST always use the SAME name, regardless of which batch you analyze.

Respond in this exact JSON format:
{
    "themes": [
        {
            "theme": "short descriptive name (2-4 words, lowercase)",
            "sentiment": "positive" or "negative",
            "mention_count": number,
            "sample_reviews": ["exact quote 1", "exact quote 2"],
            "confidence": 0.0 to 1.0
        }
    ],
    "total_reviews_analyzed": number,
    "reviews_with_no_clear_theme": number
}"""


def extract_themes_from_batch(reviews: list[dict], min_sample: int = 3) -> dict:
    """
    Send a batch of text reviews to the LLM for theme extraction.

    This is where the LLM call happens. Notice the structure:
    - System prompt: defines the role and rules (constant)
    - User prompt: contains the actual reviews (changes per batch)

    Why batch? Because sending 10,000 reviews at once would:
    1. Exceed the context window (model can only "see" so much text)
    2. Cost more money (you pay per token)
    3. Produce worse results (too much noise, model loses focus)

    Typical batch size: 100-200 reviews. Enough signal, manageable context.
    """

    # Build the user prompt with the actual review data
    # Only include reviews that have text (ratings-only reviews have no themes)
    text_reviews = [r for r in reviews if r.get("text") and len(r["text"].strip()) > 3]

    if not text_reviews:
        return {"themes": [], "total_reviews_analyzed": 0, "reviews_with_no_clear_theme": 0}

    # Format reviews for the LLM
    # We include rating + text so the model knows the sentiment context
    formatted = []
    for i, r in enumerate(text_reviews[:200]):  # Cap at 200 per batch
        formatted.append(f"[Review {i+1}] Rating: {r['rating']}/5 | \"{r['text'][:500]}\"")

    user_prompt = f"""Analyze these {len(formatted)} app reviews and extract the main themes.
Minimum sample size for a theme to be reported: {min_sample} mentions.

REVIEWS:
{chr(10).join(formatted)}"""

    print(f"    Sending {len(formatted)} reviews to LLM for theme extraction...")
    result = call_llm(
        system_prompt=THEME_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,  # Low temperature = consistent, analytical output
    )

    return result


# ============================================================
# PART 3: Orchestration — month-by-month processing
# ============================================================

def get_month_ranges(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    """
    Split a date range into monthly chunks.
    Returns list of (period_label, month_start, month_end).

    e.g., "2025-06-01" to "2025-08-31" returns:
        [("2025-06", "2025-06-01", "2025-06-30"),
         ("2025-07", "2025-07-01", "2025-07-31"),
         ("2025-08", "2025-08-01", "2025-08-31")]
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    months = []

    current = start.replace(day=1)
    while current <= end:
        month_start = current.strftime("%Y-%m-%d")
        # Get last day of month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        month_end = (next_month - timedelta(days=1)).strftime("%Y-%m-%d")
        label = current.strftime("%Y-%m")

        months.append((label, month_start, month_end))
        current = next_month

    return months


def process_period(app_id: str, period_type: str, period_label: str,
                   start_date: str, end_date: str) -> dict:
    """
    Full analysis pipeline for a single period.
    This is the core loop that runs for each month (or week, or quarter).

    Steps:
        1. Pull reviews from database for this period
        2. Compute rating statistics (pure math)
        3. Extract themes via LLM (AI analysis)
        4. Filter themes by statistical significance
        5. Store everything back to database
    """

    print(f"\n  Processing {period_type}: {period_label} ({start_date} to {end_date})")

    # Step 1: Get reviews for this period
    reviews = get_reviews_for_period(app_id, start_date, end_date)
    print(f"    Found {len(reviews)} reviews")

    if not reviews:
        print(f"    No reviews for this period. Skipping.")
        return {}

    # Step 2: Rating statistics (no LLM needed — pure math)
    stats = compute_rating_stats(reviews)
    stats.update({
        "period_type": period_type,
        "period_label": period_label,
        "period_start": start_date,
        "period_end": end_date,
    })
    store_period_analysis(app_id, stats)
    print(f"    Rating stats: avg={stats['avg_rating']}, total={stats['total_reviews']}")

    # Step 3: Theme extraction (LLM-powered)
    text_reviews = [r for r in reviews if r.get("text") and len(r["text"].strip()) > 3]

    themes_data = {"themes": []}
    if text_reviews:
        themes_data = extract_themes_from_batch(reviews)

    # Step 4: Filter by statistical significance
    significant_themes = []
    for theme in themes_data.get("themes", []):
        if check_statistical_significance(
            theme.get("mention_count", 0),
            len(text_reviews)
        ):
            significant_themes.append(theme)
        else:
            print(f"    Dropped theme '{theme.get('theme')}' — insufficient sample size "
                  f"({theme.get('mention_count', 0)} mentions)")

    # Step 5: Store themes
    if significant_themes:
        store_themes(app_id, period_type, period_label, significant_themes)
        pos = [t for t in significant_themes if t["sentiment"] == "positive"]
        neg = [t for t in significant_themes if t["sentiment"] == "negative"]
        print(f"    Themes stored: {len(pos)} positive, {len(neg)} negative")

    return {
        "stats": stats,
        "themes": significant_themes,
    }


def get_year_ranges(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    """Split a date range into yearly chunks."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    years = []

    current_year = start.year
    while current_year <= end.year:
        year_start = f"{current_year}-01-01"
        year_end = f"{current_year}-12-31"
        label = str(current_year)
        years.append((label, year_start, year_end))
        current_year += 1

    return years


def get_quarter_ranges(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    """Split a date range into quarterly chunks."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    quarters = []

    current = start.replace(day=1)
    while current <= end:
        q = (current.month - 1) // 3 + 1
        q_start_month = (q - 1) * 3 + 1
        q_end_month = q * 3
        q_start = current.replace(month=q_start_month, day=1)
        if q_end_month == 12:
            q_end = datetime(current.year, 12, 31)
        else:
            q_end = datetime(current.year, q_end_month + 1, 1) - timedelta(days=1)

        label = f"{current.year}-Q{q}"
        quarters.append((label, q_start.strftime("%Y-%m-%d"), q_end.strftime("%Y-%m-%d")))

        # Move to next quarter
        if q_end_month >= 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, q_end_month + 1, 1)

    return quarters


def get_week_ranges(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    """Split a date range into weekly chunks (Sunday to Saturday)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    weeks = []

    # Align to the nearest Sunday
    days_since_sunday = (start.weekday() + 1) % 7
    current = start - timedelta(days=days_since_sunday)

    while current <= end:
        week_start = current
        week_end = current + timedelta(days=6)  # Saturday
        label = f"W{week_start.strftime('%Y-%m-%d')}"
        weeks.append((label, week_start.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d")))
        current += timedelta(days=7)

    return weeks


def process_period_stats_only(app_id: str, period_type: str, period_label: str,
                               start_date: str, end_date: str) -> dict:
    """
    Compute only rating statistics for a period (no LLM call).
    Used for quarterly/yearly where themes are aggregated from monthly.
    """
    reviews = get_reviews_for_period(app_id, start_date, end_date)
    if not reviews:
        return {}

    stats = compute_rating_stats(reviews)
    stats.update({
        "period_type": period_type,
        "period_label": period_label,
        "period_start": start_date,
        "period_end": end_date,
    })
    store_period_analysis(app_id, stats)
    return stats


def run_analysis(app_id: str, start_date: str, end_date: str,
                 force_rerun: bool = False,
                 progress_callback=None) -> dict:
    """
    Main analysis entry point.

    Smart analysis:
    - Checks which months are already analyzed
    - Only processes new/unanalyzed months with LLM
    - Quarterly and yearly themes are AGGREGATED from monthly (no extra LLM calls)
    - force_rerun=True skips the duplicate check and re-analyzes everything

    Args:
        progress_callback: optional function(current_step, total_steps, message)
    """

    print("=" * 60)
    print(f"ANALYSIS: {app_id}")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 60)

    # Determine which months need analysis
    months = get_month_ranges(start_date, end_date)
    already_analyzed = set(get_analyzed_months(app_id)) if not force_rerun else set()

    months_to_analyze = [(l, s, e) for l, s, e in months if l not in already_analyzed]
    months_skipped = len(months) - len(months_to_analyze)

    if months_skipped > 0:
        print(f"Skipping {months_skipped} already-analyzed months")

    quarters = get_quarter_ranges(start_date, end_date)
    years = get_year_ranges(start_date, end_date)

    # Total steps: new months (LLM) + all quarters (stats) + all years (stats)
    total_steps = len(months_to_analyze) + len(quarters) + len(years)
    current_step = 0

    # ---- Monthly: full LLM analysis only for new months ----
    for i, (label, m_start, m_end) in enumerate(months_to_analyze):
        current_step += 1
        msg = f"Monthly: {label} ({current_step}/{total_steps})"
        print(f"\n  {msg}")
        if progress_callback:
            progress_callback(current_step, total_steps, msg)
        process_period(app_id, "monthly", label, m_start, m_end)

    # ---- Quarterly: stats only + aggregate themes from monthly ----
    for i, (label, q_start, q_end) in enumerate(quarters):
        current_step += 1
        msg = f"Quarterly: {label} ({current_step}/{total_steps})"
        print(f"\n  {msg}")
        if progress_callback:
            progress_callback(current_step, total_steps, msg)

        process_period_stats_only(app_id, "quarterly", label, q_start, q_end)

        # Aggregate themes from the monthly analyses within this quarter
        q_months = [m_label for m_label, m_start_d, m_end_d in months
                     if m_start_d >= q_start and m_end_d <= q_end]
        agg_themes = aggregate_themes_from_monthly(app_id, "quarterly", label, q_months)
        if agg_themes:
            store_themes(app_id, "quarterly", label, agg_themes)

    # ---- Yearly: stats only + aggregate themes from monthly ----
    for i, (label, y_start, y_end) in enumerate(years):
        current_step += 1
        msg = f"Yearly: {label} ({current_step}/{total_steps})"
        print(f"\n  {msg}")
        if progress_callback:
            progress_callback(current_step, total_steps, msg)

        process_period_stats_only(app_id, "yearly", label, y_start, y_end)

        # Aggregate themes from all monthly analyses within this year
        y_months = [m_label for m_label, m_start_d, m_end_d in months
                     if m_start_d >= y_start and m_end_d <= y_end]
        agg_themes = aggregate_themes_from_monthly(app_id, "yearly", label, y_months)
        if agg_themes:
            store_themes(app_id, "yearly", label, agg_themes)

    # Update metadata
    update_metadata(app_id, "last_analyzed_date", end_date)
    update_metadata(app_id, "seagull_analysis_complete", "true")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Analyzed {len(months_to_analyze)} new months (skipped {months_skipped})")
    print(f"Aggregated {len(quarters)} quarters, {len(years)} years")
    print("=" * 60)

    return {
        "months_analyzed": len(months_to_analyze),
        "months_skipped": months_skipped,
        "quarters": len(quarters),
        "years": len(years),
    }


# Quick test
if __name__ == "__main__":
    run_seagull_analysis(
        app_id="com.spotify.music",
        start_date="2026-01-01",
        end_date="2026-02-12",
    )

"""
Review scraper — Agent 1's first tool.
Fetches reviews from Google Play Store and Apple App Store.
"""

from datetime import datetime, timezone
from typing import Optional
from google_play_scraper import Sort, reviews, app as gplay_app
import requests

from app.models import Review, AppInfo


def scrape_google_play(app_id: str, count: int = 10000,
                       since_date: datetime = None, until_date: datetime = None) -> tuple[AppInfo, list[Review]]:
    """
    Fetch reviews from Google Play Store.

    Args:
        app_id:     The app's package name (e.g., "com.spotify.music").
        count:      Maximum number of reviews to fetch.
        since_date: Only keep reviews on or after this date. Also stops fetching
                    when all reviews in a batch are older than this date (saves API calls).
        until_date: Only keep reviews on or before this date.

    Returns:
        A tuple of (app info, list of reviews within the date range).
    """

    # Step 1: Get app info (name, etc.)
    print(f"Fetching app info for: {app_id}")
    app_details = gplay_app(app_id)
    app_info = AppInfo(
        app_id=app_id,
        app_name=app_details.get("title", "Unknown"),
        store="google_play"
    )
    print(f"App found: {app_info.app_name}")

    # Step 2: Fetch reviews in batches
    all_reviews = []
    continuation_token = None
    batch_number = 0
    hit_date_boundary = False

    date_info = ""
    if since_date:
        date_info += f" since {since_date.strftime('%Y-%m-%d')}"
    if until_date:
        date_info += f" until {until_date.strftime('%Y-%m-%d')}"
    print(f"Fetching up to {count} reviews{date_info}...")

    while len(all_reviews) < count and not hit_date_boundary:
        batch_number += 1

        result, continuation_token = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=min(200, count - len(all_reviews)),
            continuation_token=continuation_token
        )

        if not result:
            print(f"No more reviews available. Stopped at {len(all_reviews)} reviews.")
            break

        older_than_range = 0
        for raw in result:
            review_date = raw["at"]  # datetime object (may be timezone-aware)

            # Strip timezone info to avoid TypeError when comparing
            # google-play-scraper returns aware datetimes, our dates are naive
            if hasattr(review_date, 'tzinfo') and review_date.tzinfo is not None:
                review_date = review_date.replace(tzinfo=None)

            # If the review is older than our start date, count it
            if since_date and review_date < since_date:
                older_than_range += 1
                continue

            # If the review is newer than our end date, skip it
            if until_date and review_date > until_date:
                continue

            review = Review(
                review_id=raw["reviewId"],
                source="google_play",
                app_id=app_id,
                rating=raw["score"],
                text=raw.get("content"),
                date=review_date,  # already timezone-stripped above
                username=raw.get("userName", "Anonymous"),
                thumbs_up=raw.get("thumbsUpCount", 0)
            )
            all_reviews.append(review)

            if len(all_reviews) >= count:
                break

        print(f"  Batch {batch_number}: fetched {len(result)} reviews, "
              f"{older_than_range} older than range (kept: {len(all_reviews)})")

        # If most of this batch was older than our range, stop — we've gone past it
        if since_date and older_than_range > len(result) * 0.8:
            print(f"  Most reviews now older than {since_date.strftime('%Y-%m-%d')}. Stopping.")
            hit_date_boundary = True

        if continuation_token is None:
            break

    print(f"Done. Total reviews in date range: {len(all_reviews)}")
    all_reviews.sort(key=lambda r: r.date)

    return app_info, all_reviews


def scrape_apple_app_store(app_id: str, app_name: str, count: int = 10000) -> tuple[AppInfo, list[Review]]:
    """
    Fetch reviews from Apple App Store using the public iTunes RSS API.

    Args:
        app_id:   The numeric app ID (found in the App Store URL).
        app_name: The app name (Apple API doesn't always return this).
        count:    Maximum number of reviews to fetch.

    Note:
        Apple's public RSS API only returns the most recent ~500 reviews.
        For deeper historical data, you'd need the App Store Connect API
        (requires developer account). For our purposes, this is sufficient
        for the street view agent. Seagull view will primarily use Google Play.
    """

    app_info = AppInfo(
        app_id=app_id,
        app_name=app_name,
        store="apple_app_store"
    )

    all_reviews = []
    # Apple's RSS feed gives 50 reviews per page, up to 10 pages
    max_pages = min(10, (count // 50) + 1)

    print(f"Fetching Apple App Store reviews for: {app_name} (ID: {app_id})")

    for page in range(1, max_pages + 1):
        url = f"https://itunes.apple.com/us/rss/customerreviews/id={app_id}/page={page}/sortby=mostrecent/json"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raises exception if HTTP error (404, 500, etc.)
            data = response.json()
        except requests.RequestException as e:
            print(f"  Error fetching page {page}: {e}")
            break

        # Navigate the nested JSON structure Apple returns
        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            break

        for entry in entries:
            # Skip the first entry if it's app metadata (not a review)
            if "im:rating" not in entry:
                continue

            review = Review(
                review_id=entry.get("id", {}).get("label", f"apple_{page}_{len(all_reviews)}"),
                source="apple_app_store",
                app_id=app_id,
                rating=int(entry.get("im:rating", {}).get("label", 0)),
                text=entry.get("content", {}).get("label"),
                date=datetime.strptime(
                    entry.get("updated", {}).get("label", "")[:10],
                    "%Y-%m-%d"
                ) if entry.get("updated", {}).get("label") else datetime.now(),
                username=entry.get("author", {}).get("name", {}).get("label", "Anonymous"),
                thumbs_up=int(entry.get("im:voteSum", {}).get("label", 0))
            )
            all_reviews.append(review)

        print(f"  Page {page}: fetched {len(entries)} entries (total reviews: {len(all_reviews)})")

    print(f"Done. Total Apple reviews fetched: {len(all_reviews)}")
    all_reviews.sort(key=lambda r: r.date)

    return app_info, all_reviews


# ---- Quick test ----
# This block only runs when you execute this file directly (not when imported)
if __name__ == "__main__":
    # Test with a well-known app
    print("=" * 60)
    print("TESTING: Google Play scraper")
    print("=" * 60)
    app_info, review_list = scrape_google_play("com.spotify.music", count=50)
    print(f"\nApp: {app_info.app_name}")
    print(f"Reviews fetched: {len(review_list)}")
    if review_list:
        sample = review_list[0]
        print(f"\nSample review:")
        print(f"  Rating: {sample.rating}/5")
        print(f"  Date: {sample.date}")
        print(f"  Text: {sample.text[:100] if sample.text else '(no text)'}...")

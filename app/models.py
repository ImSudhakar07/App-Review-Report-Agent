"""
Data models — the structure of our data.
Every review, no matter which store it comes from, gets converted into these shapes.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Review:
    """A single user review from any app store."""
    review_id: str
    source: str                 # "google_play" or "apple_app_store"
    app_id: str                 # e.g., "com.spotify.music"
    rating: int                 # 1 to 5 stars
    text: Optional[str]         # Review text (None if user just gave stars)
    date: datetime
    username: str
    thumbs_up: int = 0


@dataclass
class AppInfo:
    """Basic info about the app being analyzed."""
    app_id: str
    app_name: str
    store: str                  # "google_play" or "apple_app_store"

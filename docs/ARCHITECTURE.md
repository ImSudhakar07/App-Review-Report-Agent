# App Store Agent — Architecture

**Built by Sudhakar.G**

## What is this?

App Store Agent is an AI-powered review intelligence tool that analyzes app store reviews for any app on Google Play Store or Apple App Store. It scrapes reviews, runs AI-driven sentiment and theme analysis, and presents actionable insights through an interactive dashboard with a conversational chatbot.

## Why it matters

Product teams drown in thousands of user reviews. Manually reading them is impossible at scale. This agent automates the entire pipeline — from raw reviews to structured insights — so PMs, growth teams, and developers can understand customer sentiment, track trends, and make data-driven decisions in minutes instead of days.

---

## Core AI concepts demonstrated

### 1. AI agent architecture

This is not a simple LLM wrapper. It's a full **AI agent** that:
- **Plans** — determines which periods need analysis, what's already done
- **Uses tools** — scraper, database, LLM, file writer
- **Observes** — checks for duplicates, validates statistical significance
- **Decides** — skips already-analyzed periods, filters noise from signal

### 2. Context engineering

The most critical skill in AI engineering. This agent demonstrates:
- **Batching** — reviews are processed month-by-month to avoid context window overflow
- **Filtering** — only reviews with text go to the LLM; pure ratings are handled by code
- **Structured prompts** — the LLM receives precise instructions with output format constraints
- **Noise reduction** — statistical significance filters remove themes without enough evidence

### 3. RAG (Retrieval-Augmented Generation)

The chatbot uses RAG to answer questions about review data:
1. **Retrieve** — pulls relevant period analyses and themes from the SQLite database
2. **Augment** — injects that data into the LLM prompt
3. **Generate** — produces grounded answers citing specific numbers and customer quotes

### 4. Smart incremental processing

- First analysis processes the full historical period
- Subsequent runs only analyze new/unanalyzed months (no duplicate LLM calls)
- Quarterly and yearly themes are **aggregated from monthly results** (no extra LLM cost)
- Review scraping detects and skips duplicates automatically

---

## Architecture diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        APP STORE AGENT                           │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌─────────────┐               │
│  │  Scraper  │───▶│ Database │───▶│  Processor   │              │
│  │          │    │ (SQLite) │    │  (LLM + Stats)│              │
│  │ • Google  │    │          │    │              │              │
│  │   Play    │    │ Per-app  │    │ • Rating     │              │
│  │ • Apple   │    │ isolation│    │   statistics  │              │
│  │   App     │    │          │    │ • Theme      │              │
│  │   Store   │    │ Tables:  │    │   extraction  │              │
│  └──────────┘    │ • reviews │    │   (xAI/Grok)  │              │
│                  │ • period  │    │ • Statistical │              │
│                  │   analysis│    │   significance│              │
│                  │ • themes  │◀───│   filtering   │              │
│                  │ • metadata│    └─────────────┘              │
│                  └────┬─────┘                                   │
│                       │                                          │
│            ┌──────────┴──────────┐                              │
│            │                     │                              │
│     ┌──────▼──────┐    ┌────────▼────────┐                     │
│     │  Dashboard   │    │    Chatbot      │                     │
│     │  (Streamlit) │    │    (RAG Q&A)    │                     │
│     │              │    │                 │                     │
│     │ • Rating     │    │ Retrieve data   │                     │
│     │   charts     │    │ from DB ──▶     │                     │
│     │ • Star       │    │ Augment prompt  │                     │
│     │   trends     │    │ ──▶ Generate    │                     │
│     │ • Theme      │    │ answer via LLM  │                     │
│     │   trendlines │    │                 │                     │
│     │ • Period     │    │ "How was        │                     │
│     │   detail     │    │  sentiment in   │                     │
│     └─────────────┘    │  Q4 2025?"      │                     │
│                         └─────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
```

## Data flow

```
User selects app
       │
       ▼
   Scrape reviews (Google Play / Apple App Store)
       │
       ▼
   Store in SQLite (per-app database, duplicates auto-skipped)
       │
       ▼
   Analyze month-by-month:
       ├── Rating statistics (pure Python — no LLM cost)
       │     • Count per star (1-5)
       │     • Average rating
       │     • Reviews with/without text
       │
       └── Theme extraction (LLM-powered)
             • Send text reviews in batches
             • Extract positive/negative themes
             • Filter by statistical significance
             • Store customer quotes as evidence
       │
       ▼
   Aggregate quarterly & yearly (from monthly — no extra LLM calls)
       │
       ▼
   Display on dashboard + answer questions via chatbot
```

## Smart processing logic

```
On "Run Analysis":
    1. Check which months are already analyzed
    2. Only send NEW months to the LLM (saves cost and time)
    3. Compute stats for all periods (cheap — pure math)
    4. Aggregate themes from monthly → quarterly → yearly
    
On "Re-run Analysis":
    1. Clear all analysis results (keep raw reviews)
    2. Re-process everything fresh

On "Scrape Reviews":
    1. Check what's already in the database
    2. Scrape from the store
    3. INSERT OR IGNORE — duplicates auto-skipped
    4. Report: X new, Y duplicates skipped
```

---

## Project structure

```
App Review Report Agent/
├── app/                    # All application code
│   ├── config.py           # Environment and settings loader
│   ├── models.py           # Data models (Review, AppInfo)
│   ├── database.py         # SQLite database layer
│   ├── llm_client.py       # xAI/Grok API client
│   ├── scraper.py          # Google Play + Apple App Store scrapers
│   ├── processor.py        # Analysis engine (stats + LLM themes)
│   └── dashboard.py        # Streamlit web UI + chatbot
│
├── data/
│   └── processed/          # SQLite databases (one per app)
│
├── docs/
│   └── ARCHITECTURE.md     # This file
│
├── .env                    # API keys (local only, never committed)
├── .env.example            # Template for API keys
├── .gitignore              # Excludes .env, .venv, data/, etc.
└── requirements.txt        # Python dependencies
```

## Technology stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.14 | Industry standard for AI/ML |
| LLM | xAI Grok (via OpenAI-compatible API) | Fast, cost-effective |
| Database | SQLite | Zero-config, file-based, built into Python |
| Dashboard | Streamlit | Python-native web apps, no frontend code needed |
| Charts | Plotly | Interactive, professional visualizations |
| Scraping | google-play-scraper + requests | Google Play library + custom Apple iTunes API |
| Stats | scipy, pandas | Statistical significance testing, data manipulation |

## Key design decisions

1. **Per-app database isolation** — each app gets its own `.db` file. No cross-contamination.
2. **Month-by-month processing** — avoids LLM context window overflow and enables incremental analysis.
3. **Stats vs AI split** — rating counts use code (free, fast). Theme extraction uses LLM (intelligent, costly). Never use AI for what arithmetic can do.
4. **Aggregation over re-analysis** — quarterly/yearly themes are built from monthly results, not re-computed. Saves LLM cost and preserves statistical accuracy.
5. **Statistical significance filtering** — themes with insufficient evidence are dropped, not reported. The agent never makes things up.

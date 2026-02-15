# App Review Report Agent

AI-powered review intelligence tool that scrapes, analyzes, and visualizes app store reviews. Built with Python, Streamlit, and xAI Grok.

Built by **Sudhakar.G**

---

## What it does

App Store Agent automates the entire pipeline from raw app reviews to structured, actionable insights:

1. **Scrapes** reviews from Google Play Store and Apple App Store
2. **Analyzes** sentiment and themes using AI (xAI Grok) and statistical methods
3. **Visualizes** trends through an interactive Streamlit dashboard
4. **Answers questions** about review data via a RAG-powered chatbot

## Features

- **Multi-store support** — Google Play Store and Apple App Store
- **AI theme extraction** — Identifies positive and negative themes with customer quotes as evidence
- **Smart incremental processing** — Only analyzes new months; skips already-processed data
- **Statistical significance filtering** — Drops themes without enough evidence
- **Interactive dashboard** — Rating distributions, star trends, volume charts, theme trendlines
- **RAG chatbot** — Ask natural-language questions about review data and get grounded answers
- **Per-app database isolation** — Each app gets its own SQLite file, no cross-contamination
- **Cost-efficient** — Rating stats use pure code; only theme extraction calls the LLM. Quarterly/yearly aggregations are built from monthly results with zero extra LLM cost

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| LLM | xAI Grok (via OpenAI-compatible API) |
| Database | SQLite (zero-config, file-based) |
| Dashboard | Streamlit |
| Charts | Plotly |
| Scraping | google-play-scraper + Apple iTunes API |
| Statistics | scipy, pandas |

## Getting Started

### Prerequisites

- Python 3.10+
- An [xAI API key](https://console.x.ai/)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/App-Review-Report-Agent.git
   cd App-Review-Report-Agent
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   ```

   Activate it:

   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Copy the example file and fill in your keys:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your values:

   ```
   API_KEY=your_xai_api_key_here
   DASHBOARD_USERNAME=admin
   DASHBOARD_PASSWORD=your_password_here
   ```

### Running the App

```bash
streamlit run app/dashboard.py
```

The dashboard will open in your browser. Sign in with the credentials you set in `.env`.

## How to Use

1. **Add an app** — Enter the app ID (e.g., `com.spotify.music` for Google Play, or the numeric ID for Apple App Store)
2. **Scrape reviews** — Select a date range and fetch reviews from the store
3. **Run AI analysis** — The agent processes reviews month-by-month, extracting themes and computing statistics
4. **Explore the dashboard** — View rating trends, theme trendlines, and period details
5. **Ask the chatbot** — Get answers about review data in natural language

## Project Structure

```
App Review Report Agent/
├── app/
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
│   └── ARCHITECTURE.md     # Detailed architecture documentation
│
├── .env.example            # Template for API keys
├── .gitignore              # Excludes .env, .venv, data/, etc.
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Architecture

For a deep dive into the architecture, design decisions, and data flow, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

This project is open source and available under the [MIT License](LICENSE).

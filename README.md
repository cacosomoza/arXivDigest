# arXiv Digest — Personalized Paper Ranker

A self-hosted Flask application that fetches fresh arXiv preprints and ranks them against your research profile using a hybrid TF-IDF + keyword + LLM scoring architecture. Designed for researchers who need to stay on top of their field without drowning in hundreds of daily submissions.

## Features

- **Three-way hybrid ranking**: TF-IDF cosine similarity + keyword matching + LLM semantic scoring
- **Expert curation with insights**: LLM-curated per-category digests with intelligent explanations
- **Expert ranking modes**: 7 sort modes including simple (LLM-first / Keywords-first) and advanced permutations
- **Dynamic promotion**: Extended candidates can rise into the curated list based on ranking mode
- **Batch LLM mode**: Two-pass scoring (lightweight filter + deep scoring) for speed and cost efficiency
- **RSS & API modes**: Fetch via arXiv RSS (daily) or arXiv API (date ranges)
- **Category filtering**: Filter tiered digest by insight categories with unified curated view
- **Live reblend**: Interactively adjust score weights without re-fetching or re-calling the LLM
- **Download as HTML**: Self-contained interactive digest for offline reading with full ranking controls
- **Dark mode**: Toggle between light and dark themes
- **Replacement tracking**: Detect and optionally hide arXiv paper replacements

## Quick Start (5 minutes)

### Prerequisites

- Python 3.11 or 3.12
- pip package manager
- An LLM API key (optional — TF-IDF + keyword works without one)

### Installation

```bash
# 1. Clone or extract this repository
cd arxiv-digest

# 2. Create a virtual environment (recommended)
python -m venv venv

# 3. Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
python app.py

# 6. Open in browser
# http://localhost:5000
```

### LLM API Key Setup (Optional but Recommended)

The app works without an LLM key using TF-IDF + keyword scoring only. For LLM-enhanced curation:

1. **Moonshot AI (Kimi)**: Get key from [platform.moonshot.cn](https://platform.moonshot.cn) or [platform.moonshot.ai](https://platform.moonshot.ai)
2. **OpenAI**: Get key from [platform.openai.com](https://platform.openai.com)
3. **Anthropic (Claude)**: Get key from [console.anthropic.com](https://console.anthropic.com)

Enter your key in the **Settings** tab, select your provider, and click **Test API**.

## Usage Guide

### 1. Set Up Your Profile

Navigate to the **Profile** tab and enter:
- **Keywords**: Your research interests (comma-separated, ~50-100 recommended)
- **Publications**: Your papers for TF-IDF profile vectorization
- **Categories**: arXiv categories to monitor (e.g., `quant-ph`, `physics.chem-ph`)

### 2. Define Insight Categories

In the **Settings** tab, configure topical categories for the Insights feature:
- Default categories: Quantum Chemistry, Electrochemistry, Open Quantum Systems, Quantum PDE Solvers
- Click **+ Add Category** to add more (up to 26 total)
- Each category needs: Domain Name + Keywords (comma-separated) + Penalising Keywords

### 3. Run a Digest

In the **Digest** tab:
1. Select date range or use RSS mode (today's submissions)
2. Choose LLM mode: **Batch filter + deep** (recommended), **Individual scoring**, or **No LLM**
3. Click **Fetch & Rank**
4. Papers appear tiered: Tier 1 (Must Read) through Tier 5 (Extended List) + LLM Rescue

### 4. Generate Insights

After fetching, click **Generate Insights** in the **Insights** tab:
- The LLM curates papers per category with explanations
- Set min/max papers per category in Settings
- Generation Details show LLM-selected vs keyword-backfilled counts

### 5. Expert Ranking

When viewing a category-filtered digest, use the **Expert Ranking** panel:
- **Simple**: Prioritize LLM (default) or Prioritize Keywords
- **Advanced**: 5 permutation modes + Combined alpha-beta-gamma weighted blend
- Papers promoted from Extended Candidates into the curated list are marked with a badge

### 6. Filter by Category

Use the category filter bar in the **Digest** tab to show only papers from a specific insight category. The filtered view shows the unified expert-curated list with inline LLM comments.

### 7. Download

Click **Download as HTML** to save a self-contained interactive digest that works offline. The downloaded file preserves all ranking controls and category filtering.

## Project Structure

```
arxiv-digest/
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html             # Main application UI
├── README.md                  # This file
└── TECHNICAL_DOCUMENTATION.md # Inner mechanisms and parameter tuning
```

## Deployment

### Local Network (Access from Phone)

```bash
# Find your local IP
ifconfig  # macOS/Linux
ipconfig  # Windows

# Run with host binding
python app.py
# Or directly: flask run --host=0.0.0.0 --port=5000

# Access from phone on same WiFi
# http://YOUR_LOCAL_IP:5000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 5000 | Server port |
| `FLASK_DEBUG` | True | Debug mode |
| `SECRET_KEY` | random | Flask session secret |
| `CORS_ORIGINS` | * | Allowed origins |

## LLM Cost Estimates

| Provider | Model | Cost per 100 papers | Speed |
|----------|-------|---------------------|-------|
| Moonshot | moonshot-v1-8k | ~$0.01-0.02 | Fast |
| OpenAI | gpt-4o-mini | ~$0.005-0.01 | Fast |
| Anthropic | claude-haiku | ~$0.01-0.02 | Fast |

Costs are user-borne via their own API keys. The app server has zero LLM costs.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Tabs unresponsive | `rm -rf __pycache__` and restart |
| LLM test fails (401) | Check API key is from correct provider console |
| Empty digest | Verify arXiv categories in Profile are valid |
| Insights show wrong categories | Check keyword specificity in Settings |
| Module not found | `pip install -r requirements.txt` |
| HTML download ranking broken | Ensure browser allows popups from localhost |

## Technical Documentation

For a deep dive into the inner mechanisms — the five-stage pipeline, three-way scoring formula, tier system, LLM curation modes, keyword match quality system, and parameter tuning guide — see [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md).

## License

MIT License — free for academic and commercial use.

## Acknowledgments

Developed for the quantum chemistry/physics research community. Built with Flask, scikit-learn, and love for the scientific method.

# arXiv Digest

Personalized academic paper curation for quantum chemistry and physics.

arXiv Digest fetches live preprints from arXiv, ranks them against your research interests using a multi-signal scoring pipeline, and organises the best matches into topic-specific curated lists with LLM-generated explanations. Every paper comes directly from the arXiv API; the LLM only scores and summarises, never fabricates.

**Anti-hallucination architecture:** arXiv API handles all retrieval. The LLM receives real paper metadata and returns scores, explanations, and category assignments. No paper is ever invented.

---

## The Full Pipeline

The digest operates in **five sequential stages**. Stages 1-3 produce the global digest. Stages 4-5 produce the per-category expert-curated insights.

### Stage 1: Fetch
Papers are fetched from the arXiv API (RSS "new" submissions or a date range). Only papers from your subscribed arXiv categories are retrieved.

### Stage 2: Keyword Filter
Papers are matched against your profile keywords. Papers with zero keyword matches are discarded. This is a *coarse filter* -- the goal is speed, not precision.

### Stage 3: Global Digest Ranking
Remaining papers are scored with a **three-way weighted blend**:

```
score = (w_tfidf * tfidf_norm + w_kw * kw_norm + w_llm * llm_norm) / (w_tfidf + w_kw + w_llm)
```

- **TF-IDF** -- Statistical cosine similarity between the paper and your keyword profile + publications. Rare words (e.g. "DMRG") are weighted higher than common words (e.g. "quantum").
- **Keyword count** -- How many profile keywords appear in the paper title/abstract (normalized 0-1). Simple but effective for breadth coverage.
- **LLM score** -- Semantic relevance rated by a large language model on a 0.5-1.0 scale (re-normalised to 0-1). Captures conceptual relevance that keyword matching misses.

**Default weights:** 25% TF-IDF / 25% Keyword / 50% LLM. After generation, use the **Live Reblend** sliders to interactively re-rank without re-fetching or re-calling the LLM.

**Result Tiers:**

| Tier | Score Range | Label |
|------|-------------|-------|
| Tier 1 | >= 0.80 | MUST READ |
| Tier 2 | 0.60 - 0.80 | HIGHLY RELEVANT |
| Tier 3 | 0.40 - 0.60 | RELEVANT |
| Tier 4 | 0.20 - 0.40 | POSSIBLY RELEVANT |
| Tier 5 | < 0.20 | EXTENDED LIST |

**LLM Rescue:** If the LLM scores a paper >= 0.60 but it falls outside Tiers 1-3, it is "rescued" into a special LLM Rescue tier. This catches semantically relevant papers that keyword metrics undervalue.

### Stage 4: Insights Keyword Pre-Filter
Top-ranked papers are matched against *each category's own keyword list* (independent of the profile). Each match receives a quality score. Papers below the quality threshold are discarded for that category.

### Stage 5: LLM Expert Curation
The LLM scores *all* remaining candidates 0-10 using a domain-specific expert prompt. Papers are sorted by the chosen Expert Ranking Mode. The top **max_papers** become the Expert Curated list; the rest become Extended Candidates.

**Key point:** A paper can rank #1 in the global digest but receive zero Insight matches if its keywords don't overlap with any category's list. Conversely, a low-ranked global paper can appear in Insights if it strongly matches a category. The two systems are orthogonal by design.

---

## LLM Scoring Modes

Three modes control how the LLM scores papers. Choose based on speed, cost, and precision needs.

### Individual scoring (precise, slow)
Every keyword-matched paper is sent to the LLM individually with full title + abstract. Most accurate but slowest and most expensive. Best for small date ranges or when precision is paramount.

### Batch filter + deep (fast, recommended)
Two-pass approach: (1) The LLM receives a lightweight batch (titles only) and filters out clearly irrelevant papers. (2) Surviving papers are scored in-depth with full abstracts. The **Batch filter pick ratio** controls what fraction of candidates survive pass 1 (default: 80%).

### Chunked deep scoring
When enabled, the deep-scoring pass splits candidates into chunks (max 30 per chunk) to avoid context-window limits. Useful when many papers survive the batch filter. Adds modest latency but prevents truncation.

**LLM Input Mode:** "Titles + Abstracts" (default) sends both for maximum accuracy. "Titles only" is faster but less precise. "Abstracts only" avoids title-bias. "Structured JSON" returns machine-readable scores for downstream processing.

---

## Expert Ranking Modes

Six permutation modes and one combined mode determine how papers are ordered within each category. **All** candidates (curated + extended) are sorted together; the top **max_papers** become the Expert Curated list. Papers from the extended list that rise above the cutoff are marked with a **Promoted** badge.

| Mode | Primary | Tie-break | Final |
|------|---------|-----------|-------|
| LLM -> Digest -> QS | LLM expert score | Digest score | Quality score |
| LLM -> QS -> Digest | LLM expert score | Quality score | Digest score |
| Digest -> LLM -> QS | Digest score | LLM expert score | Quality score |
| Digest -> QS -> LLM | Digest score | Quality score | LLM expert score |
| QS -> LLM -> Digest | Quality score | LLM expert score | Digest score |
| QS -> Digest -> LLM | Quality score | Digest score | LLM expert score |
| Combined alpha-beta-gamma | Weighted sum: alpha*LLM + beta*Digest + gamma*QS | -- | -- |

All three scores are min-max normalised (0-1) before blending in the combined mode.

**Dynamic promotion:** When you switch ranking modes, papers from the Extended Candidates list can rise above the max_papers cutoff and enter the Expert Curated list. They are marked with a **Promoted** badge. This is not a new selection -- it is a re-ranking of the same candidate pool.

---

## Keyword Match Quality System

Not all keyword matches are equal. The system assigns **quality points** based on match precision:

| Match type | Example | Quality | Force-include? |
|------------|---------|---------|----------------|
| L1: Full phrase | "heat equation" appears exactly | 2.0 | Yes (QS >= 2.0) |
| L3: Acronym | "PDE" from "partial differential equations" | 1.5 | Yes (QS >= 1.5) |
| L4: Word-bag | All individual words present (any order) | 1.0 | Yes (QS >= 1.5) |

**Force-include thresholds:** After LLM curation, any paper whose total keyword quality score meets the threshold is force-added to the curated list, even if the LLM scored it low. This prevents the LLM from overlooking papers with strong keyword evidence.

**Note:** Sub-phrase matching (L2) was removed because it caused false positives (e.g. "time evolution" incorrectly matching "imaginary time evolution"). If you want both, add both as separate keywords.

### Keyword conventions

| You list | Also catches | You DON'T need |
|----------|-------------|----------------|
| exciton | excitons, Exciton, Excitons | excitons |
| non-linear | nonlinear, Non-Linear | nonlinear |
| Excited States | excited states | lowercase variant |

| You MUST list both | Why |
|-------------------|-----|
| cavity + cavities | -ies plurals not handled automatically |
| time evolution + imaginary time evolution | No sub-phrase matching (removed for precision) |

---

## Replacement Tracking

arXiv allows authors to replace previously submitted papers. The digest detects replacements via the `<dc:type>replace</dc:type>` RSS field and marks them with a **Replacement** badge. You can toggle replacements on or off via the "Hide replacements" / "Show replacements" button.

**Why hide replacements?** If you read the original submission, the replacement may contain only minor corrections. Hiding them declutters the digest. If you missed the original, keep them visible.

---

## Settings Reference

### LLM Provider
Which API to call for LLM scoring. Each provider offers different models, pricing, and latency. You only need an API key for the selected provider.

| Provider | Notes |
|----------|-------|
| Moonshot AI (Kimi k1.5) | Chinese provider, strong Chinese-English bilingual capability |
| OpenAI (GPT-4o-mini) | Fast, cost-effective, good at structured JSON output |
| Anthropic (Claude Haiku) | Fast, strong scientific reasoning, good at following complex prompts (default) |

### LLM Mode
Controls the LLM scoring strategy.

| Mode | Description |
|------|-------------|
| Individual scoring | One API call per paper. Most accurate, slowest. |
| Batch filter + deep | Two-pass: lightweight filter then deep scoring. Fast, recommended (default). |

### Batch filter pick ratio (default: 80%)
In Batch filter mode, this is the fraction of candidates that survive the lightweight filter pass and proceed to deep scoring. Higher = more papers scored deeply, more API cost. Lower = faster, but may miss borderline papers.

| Value | Effect |
|-------|--------|
| 50% | Aggressive filtering. Fastest, cheapest. Risk of false negatives. |
| 80% | Balanced. Good coverage without excessive API calls (default). |
| 100% | No filtering. All candidates go to deep scoring. Maximum cost. |

### Chunked deep scoring (default: ON)
Splits deep-scoring candidates into chunks of 30 to avoid context-window limits. Recommended for large digests. Adds modest latency but prevents truncation.

### LLM Input Mode (default: Titles + Abstracts)
What text the LLM receives for each paper.

| Mode | Description |
|------|-------------|
| Titles + Abstracts | Full information. Most accurate (default). |
| Titles only | Fastest. Good for broad filtering. |
| Abstracts only | Avoids title-clickbait bias. |
| Structured JSON | Returns machine-readable scores. For automation. |

### Max LLM Candidates (default: All fetched)
Caps how many keyword-matched papers are sent to the LLM. "All fetched" sends every matched paper. Lower values speed up the digest but may miss relevant papers that keyword-matched weakly.

### Cross-discipline ratio (default: 0.1)
Controls how multi-category papers are handled. If a paper's best category has quality Q, it stays in a second category only if that category's quality is at least Q x ratio.

| Value | Effect |
|-------|--------|
| 0.1 | Very lenient. Almost all multi-category papers stay in all matching categories (default). |
| 0.5 | Balanced. Second category needs >= 50% of best quality. |
| 1.0 | Strict. Only perfectly balanced papers stay in multiple categories. |

### Min quality threshold (default: 1.0)
Absolute floor for keyword quality in Insights. A category must have at least this much total quality to keep any paper. Prevents categories with only weak word-bag matches from retaining papers.

| Value | Effect |
|-------|--------|
| 0.5 | Very lenient. A single word-bag match keeps papers. |
| 1.0 | Balanced. Requires at least one strong match or equivalent mix (default). |
| 2.0 | Strict. Needs at least one full phrase match (L1) to retain any paper. |

### Min / max papers per category (default: 12 / 24)
Controls the size of each category's Expert Curated list.

- **Min:** Guarantees at least N papers. If the LLM returns fewer, keyword backfill is used.
- **Max:** Upper cap after sorting. Weakest papers are cut first.

**Per-category overrides:** Each category can have its own min/max. Electrochemistry and Quantum PDE Solvers default to 4-6 (low daily volume). All others default to 12-24.

### Match weights (default: L1=2.0, L3=1.5, L4=1.0)
Controls how much each keyword match type contributes to the quality score. Higher L1 rewards precise phrase matches. Lower L4 reduces noise from word-bag matches.

### Score weights (default: 25 / 25 / 50)
The three weights for the global digest score (TF-IDF / Keyword / LLM). Auto-normalise to sum to 100%. After generation, use Live Reblend to interactively re-rank without re-fetching.

| Configuration | Effect |
|---------------|--------|
| Keyword-heavy (50/25/25) | Favors papers with many keyword hits. Good for broad coverage. |
| Balanced (25/25/50) | Default. LLM has strongest voice; TF-IDF and keyword provide grounding. |
| LLM-only (0/0/100) | Pure semantic ranking. Risk of missing papers the LLM undervalues. |

---

## Download

The **Download** button generates a self-contained HTML file with the complete digest, including all paper metadata, LLM explanations, category summaries, and Expert Ranking Mode controls. The file works offline and preserves the full interactive experience (expandable details, category filtering, ranking mode switching).

| Property | Value |
|----------|-------|
| File name | arxiv-digest-YYYY-MM-DD.html |
| Size | ~200-500 KB (all data embedded, no external dependencies) |
| Contents | All tiers, all insights, all explanations, search criteria, date range, and ranking mode controls |

**Tip:** Download before closing the browser tab. The file contains everything needed to reconstruct the digest view, including the full paper list and all LLM-generated content.

---

## Tuning Guide

### "Too many irrelevant papers in Insights"
Raise the **min quality threshold** (e.g. 1.5 or 2.0) to require stronger keyword evidence. Check your category keywords for overly broad terms. Add **penalising keywords** to filter out off-topic subfields.

### "My category is always empty"
Lower the **min quality threshold** (e.g. 0.5). Add broader keywords to the category. Check that the category's keywords are actually present in arXiv abstracts (not just paper titles). Increase **max LLM candidates** so more papers reach the Insights stage.

### "The same paper appears in every category"
Lower the **cross-discipline ratio** (e.g. 0.1) to be more lenient, or raise it (e.g. 0.8) to be stricter. Check for overly broad keywords shared across categories. Add more specific keywords to differentiate categories.

### "LLM is missing papers I care about"
Switch to **Individual scoring** mode for maximum LLM attention per paper. Use **Titles + Abstracts** input mode. Edit the category's **LLM prompt** to explicitly mention the subtopics you care about. Check the **force-include threshold** -- if your keywords are strong, the paper may already be force-included.

### "Digest is too expensive / slow"
Use **Batch filter + deep** mode with a lower **batch filter pick ratio** (e.g. 50%). Reduce **max LLM candidates** (e.g. 50). Use **Titles only** input mode. Narrow the **date range** or subscribe to fewer arXiv categories.

### "I want more papers from a small category"
Set a higher **per-category max** for that category (e.g. 15). Lower its **min quality threshold**. Add broader keywords. Consider adding **cross-cutting keywords** that overlap with the category's interests.

---

## Data Privacy

All processing happens on your local Flask server. API keys are sent directly from your machine to the LLM provider. No paper data, keywords, or API keys pass through any third-party server other than the LLM provider you select.

| Component | Data flow |
|-----------|-----------|
| arXiv API | Fetches public preprint metadata only |
| LLM API | Receives paper titles/abstracts + your prompts. Subject to provider's privacy policy. |
| Local storage | Profile and settings saved in browser localStorage. No cloud sync. |

---

## Installation

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

---

## License

MIT

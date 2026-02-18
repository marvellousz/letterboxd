# letterboxd movie recommender

get personalized movie recs based on your letterboxd ratings. uses criticker-style percentile matching + vibe-based semantic search.

## how it works

1. scrapes your letterboxd ratings from diary view (bypasses cloudflare)
2. matches films to movielens dataset
3. finds taste-compatible users via percentile ranking (not just popularity)
4. reranks candidates using sentence embeddings for vibe similarity
5. gives you top 5 picks

## setup

```bash
uv venv
uv pip install -r requirements.txt
```

## usage

```bash
uv run python main.py
# enter your letterboxd profile url when prompted

# or pass directly
uv run python main.py --url https://letterboxd.com/username/

# debug mode
uv run python main.py --debug --url https://letterboxd.com/username/
```

## requirements

- python 3.10+
- letterboxd profile with public diary
- at least 5 distinct ratings (for percentile calc)
- at least some overlap with movielens catalog

## how it's different

most rec systems just push popular movies. this one:
- uses **percentiles** instead of raw ratings (harsh critic vs easy rater normalized)
- calculates **taste compatibility index (tci)** like criticker
- only uses opinions from users who actually match your taste
- adds semantic layer for genre/vibe preferences

## tech stack

- cloudscraper (bypass protection)
- beautifulsoup4 (diary scraping)
- pandas (data wrangling)
- sentence-transformers (embeddings)
- movielens dataset (collaborative filtering base)

## limits

- only works for users with mainstream-ish taste (needs movielens matches)
- movielens-small has ~9k movies (not super recent releases)
- percentile calc needs 5+ distinct ratings minimum
- first run downloads ~90mb sentence transformer model

## files

- `scraper.py` - letterboxd diary scraper
- `model.py` - criticker-style percentile + tci calculations
- `recommender.py` - semantic vibe reranking
- `main.py` - pipeline orchestration

no bs, just movies you'll actually like.

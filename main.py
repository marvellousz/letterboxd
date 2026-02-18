from __future__ import annotations

import argparse
import sys
from typing import Optional

from model import train_and_recommend
from recommender import refine_with_vibe
from scraper import LetterboxdScrapeError, scrape_letterboxd_films


def run_pipeline(letterboxd_url: str, output_csv: str = "user_data.csv", debug: bool = False) -> int:
    try:
        print("[1/4] Scraping Letterboxd films...")
        user_df = scrape_letterboxd_films(letterboxd_url, output_csv=output_csv, debug=debug)
        print(f"  Collected {len(user_df)} rated films.")

        print("[2/4] Training collaborative filtering model...")
        artifacts = train_and_recommend(user_data_path=output_csv, top_k=50)
        print(f"  Matched {len(artifacts.user_history)} films to MovieLens.")

        print("[3/4] Applying vibe-based semantic reranking...")
        recommendations = refine_with_vibe(
            candidates_df=artifacts.candidates,
            user_history_df=artifacts.user_history,
            top_n=5,
        )

        print("\n[4/4] Top 5 recommendations\n")
        if not recommendations:
            print("No recommendations generated. Try another user profile.")
            return 1

        for idx, rec in enumerate(recommendations, start=1):
            year_fragment = f" ({rec.year})" if rec.year else ""
            print(f"{idx}. {rec.title}{year_fragment}")
            print(f"   Why: {rec.why}")
            print(f"   Final Score: {rec.score:.3f}\n")

        return 0

    except LetterboxdScrapeError as exc:
        print(f"Scraping error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # broad catch for user-facing CLI resilience
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Movie recommendation pipeline: scrape Letterboxd ratings, train CF model, and rerank by vibe."
        )
    )
    parser.add_argument("--url", help="Letterboxd profile URL")
    parser.add_argument("--output", default="user_data.csv", help="Path for scraped user CSV")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    letterboxd_url: Optional[str] = args.url
    if not letterboxd_url:
        letterboxd_url = input("Enter your Letterboxd profile URL: ").strip()

    if not letterboxd_url:
        print("No URL provided.", file=sys.stderr)
        return 1

    return run_pipeline(letterboxd_url=letterboxd_url, output_csv=args.output, debug=args.debug)


if __name__ == "__main__":
    raise SystemExit(main())

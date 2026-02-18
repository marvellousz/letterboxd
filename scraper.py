from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


class LetterboxdScrapeError(RuntimeError):
    """Raised when scraping Letterboxd fails."""


@dataclass
class FilmRecord:
    title: str
    year: int
    rating: float


STAR_MAP = {
    "★": 1.0,
    "★★": 2.0,
    "★★★": 3.0,
    "★★★★": 4.0,
    "★★★★★": 5.0,
    "½": 0.5,
    "★½": 1.5,
    "★★½": 2.5,
    "★★★½": 3.5,
    "★★★★½": 4.5,
}


def _normalize_films_url(profile_or_films_url: str) -> str:
    cleaned = profile_or_films_url.strip().rstrip("/")
    if not cleaned.startswith("http"):
        raise ValueError("Please provide a full Letterboxd URL (https://letterboxd.com/<user>/).")
    return cleaned if cleaned.endswith("/films") else f"{cleaned}/films"


def _extract_rating(film_node: BeautifulSoup) -> Optional[float]:
    rating_node = film_node.select_one("p.poster-viewingdata span.rating")
    if rating_node is None:
        return None
    rating_raw = rating_node.get_text(strip=True)
    return STAR_MAP.get(rating_raw)


def _extract_year(film_node: BeautifulSoup) -> Optional[int]:
    year_node = film_node.select_one("small.releasedate")
    if year_node is None:
        return None
    year_text = year_node.get_text(strip=True)
    if not re.match(r"^\d{4}$", year_text):
        return None
    return int(year_text)


def _parse_films_page(soup: BeautifulSoup) -> List[FilmRecord]:
    records: List[FilmRecord] = []
    for film in soup.select("li.poster-container"):
        title_node = film.select_one("img")
        title = title_node.get("alt", "").strip() if title_node else ""
        if not title:
            continue

        year = _extract_year(film)
        rating = _extract_rating(film)
        if year is None or rating is None:
            continue

        records.append(FilmRecord(title=title, year=year, rating=rating))
    return records


def _validate_response(response: requests.Response, target_url: str) -> None:
    if response.status_code == 404:
        raise LetterboxdScrapeError(f"Letterboxd page not found (404): {target_url}")
    if response.status_code >= 400:
        raise LetterboxdScrapeError(f"Failed to access Letterboxd URL ({response.status_code}): {target_url}")

    lower_text = response.text.lower()
    if "this member's profile is private" in lower_text:
        raise LetterboxdScrapeError("Profile is private. Please use a public Letterboxd account.")
    if "sorry, we can’t find the page" in lower_text or "sorry, we can't find the page" in lower_text:
        raise LetterboxdScrapeError(f"Letterboxd page not found: {target_url}")


def scrape_letterboxd_films(profile_or_films_url: str, output_csv: str = "user_data.csv") -> pd.DataFrame:
    """Scrape a Letterboxd user's films and ratings into a DataFrame and CSV."""
    films_url = _normalize_films_url(profile_or_films_url)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        }
    )

    all_records: List[FilmRecord] = []
    page = 1

    while True:
        page_url = f"{films_url}/page/{page}/"
        response = session.get(page_url, timeout=20)
        _validate_response(response, page_url)

        soup = BeautifulSoup(response.text, "html.parser")
        page_records = _parse_films_page(soup)
        if not page_records:
            break

        all_records.extend(page_records)
        page += 1

    if not all_records:
        raise LetterboxdScrapeError(
            "No rated films found. Ensure the profile has rated films visible on the Films page."
        )

    df = pd.DataFrame([record.__dict__ for record in all_records]).drop_duplicates(subset=["title", "year"])
    df = df.sort_values(by="rating", ascending=False).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Letterboxd films with ratings.")
    parser.add_argument("url", help="Letterboxd profile URL (or /films URL)")
    parser.add_argument("--output", default="user_data.csv", help="Output CSV path")
    args = parser.parse_args()

    result = scrape_letterboxd_films(args.url, args.output)
    print(f"Saved {len(result)} films to {args.output}")

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from requests import RequestException
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
    if not cleaned:
        raise ValueError("Please provide a Letterboxd profile URL or username.")

    if not cleaned.startswith("http"):
        username = cleaned.lstrip("@")
        if not re.match(r"^[A-Za-z0-9_\-]+$", username):
            raise ValueError("Please provide a full Letterboxd URL (https://letterboxd.com/<user>/).")
        cleaned = f"https://letterboxd.com/{username}"

    parsed = urlparse(cleaned)
    if parsed.netloc and "letterboxd.com" not in parsed.netloc:
        raise ValueError("Please provide a Letterboxd URL on letterboxd.com.")
    return cleaned if cleaned.endswith("/films") else f"{cleaned}/films"


def _build_http_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Upgrade-Insecure-Requests": "1",
        }
    )
    return session


def _extract_rating(film_node: BeautifulSoup) -> Optional[float]:
    rating_node = film_node.select_one("p.poster-viewingdata span.rating, span.rating")
    if rating_node is not None:
        rating_raw = rating_node.get_text(strip=True)
        mapped_rating = STAR_MAP.get(rating_raw)
        if mapped_rating is not None:
            return mapped_rating

        class_names = " ".join(rating_node.get("class", []))
        class_match = re.search(r"rated-(\d{1,2})", class_names)
        if class_match:
            return int(class_match.group(1)) / 2

    class_match = re.search(r"rated-(\d{1,2})", " ".join(film_node.get("class", [])))
    if class_match:
        return int(class_match.group(1)) / 2

    rating_attr_node = film_node.select_one("[data-rating]")
    if rating_attr_node is not None:
        raw_rating = rating_attr_node.get("data-rating", "").strip()
        if re.match(r"^\d+(?:\.\d+)?$", raw_rating):
            return float(raw_rating)

    return None


def _extract_year(film_node: BeautifulSoup) -> Optional[int]:
    poster_node = film_node.select_one("div.film-poster, div.poster")
    if poster_node is not None:
        for attr in ("data-film-release-year", "data-year"):
            year_raw = poster_node.get(attr)
            if year_raw and re.match(r"^\d{4}$", year_raw.strip()):
                return int(year_raw.strip())

    year_node = film_node.select_one("small.releasedate")
    if year_node is None:
        title_node = film_node.select_one("img")
        title_text = title_node.get("alt", "") if title_node else ""
        title_match = re.search(r"\((\d{4})\)\s*$", title_text)
        if title_match:
            return int(title_match.group(1))
        return None

    year_text = year_node.get_text(strip=True)
    year_match = re.search(r"(\d{4})", year_text)
    return int(year_match.group(1)) if year_match else None


def _extract_title(film_node: BeautifulSoup) -> str:
    title_node = film_node.select_one("img")
    title = title_node.get("alt", "").strip() if title_node else ""
    if not title:
        linked_poster = film_node.select_one("[data-target-link]")
        if linked_poster is not None:
            target_link = linked_poster.get("data-target-link", "")
            slug = target_link.strip("/").split("/")[-1]
            if slug:
                title = slug.replace("-", " ").title()
    if not title:
        return ""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def _parse_films_page(soup: BeautifulSoup) -> Tuple[List[FilmRecord], int]:
    films = soup.select("li.poster-container, li.posteritem")
    if not films:
        films = soup.select("ul.poster-list > li")
    records: List[FilmRecord] = []
    for film in films:
        title = _extract_title(film)
        if not title:
            continue

        year = _extract_year(film)
        rating = _extract_rating(film)
        if year is None or rating is None:
            continue

        records.append(FilmRecord(title=title, year=year, rating=rating))
    return records, len(films)


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

    session = _build_http_session()

    # Prime cookies to reduce anti-bot false positives on direct /page/1 requests.
    profile_url = f"{films_url.rsplit('/films', maxsplit=1)[0]}/"
    for warmup_url in (profile_url, f"{films_url}/"):
        try:
            warmup_response = session.get(warmup_url, timeout=20)
        except RequestException as exc:
            raise LetterboxdScrapeError(f"Network error while accessing Letterboxd: {exc}") from exc
        _validate_response(warmup_response, warmup_url)

    all_records: List[FilmRecord] = []
    page = 1

    while True:
        page_url = f"{films_url}/page/{page}/"
        try:
            response = session.get(page_url, timeout=20)
        except RequestException as exc:
            raise LetterboxdScrapeError(f"Network error while accessing Letterboxd: {exc}") from exc

        if response.status_code == 403:
            try:
                import cloudscraper
            except ImportError:
                pass
            else:
                scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "linux"})
                scraper.trust_env = False
                try:
                    response = scraper.get(page_url, timeout=20)
                except RequestException as exc:
                    raise LetterboxdScrapeError(f"Network error while accessing Letterboxd: {exc}") from exc

        _validate_response(response, page_url)

        soup = BeautifulSoup(response.text, "html.parser")
        page_records, film_count = _parse_films_page(soup)
        if film_count == 0:
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

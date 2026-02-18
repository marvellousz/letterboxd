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
    year: Optional[int]
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


def _parse_rating_value(raw_rating: str) -> Optional[float]:
    rating = raw_rating.strip()
    if not re.match(r"^\d+(?:\.\d+)?$", rating):
        return None

    value = float(rating)
    # Letterboxd frequently stores half-star units (e.g. 8 == 4.0 stars).
    return value / 2 if value > 5 else value


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
    
    # Use diary view which has ratings in HTML table format
    if "/films/diary" in cleaned:
        return cleaned
    elif "/films" in cleaned:
        return cleaned.replace("/films", "/films/diary")
    else:
        return f"{cleaned}/films/diary"


def _build_http_session():
    """Build a cloudscraper session to bypass Cloudflare protection."""
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'linux', 'mobile': False}
        )
        scraper.trust_env = False
        return scraper
    except ImportError:
        # Fallback to regular session if cloudscraper not available
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
            }
        )
        return session


def _extract_rating(film_node: BeautifulSoup) -> Optional[float]:
    """Extract rating from a film node (works for both grid and table views)."""
    # Check for rated-X class pattern (used in diary table view)
    rating_span = film_node.select_one("span.rating[class*='rated-']")
    if rating_span:
        classes = rating_span.get("class", [])
        for cls in classes:
            if cls.startswith("rated-"):
                try:
                    rating_value = int(cls.replace("rated-", ""))
                    return rating_value / 2.0  # Convert to 5-star scale
                except (ValueError, AttributeError):
                    pass
    
    # Legacy extraction methods for older page formats
    for attr in ("data-owner-rating", "data-rating"):
        raw_rating = (film_node.get(attr) or "").strip()
        parsed = _parse_rating_value(raw_rating)
        if parsed is not None:
            return parsed

    for selector in ("[data-owner-rating]", "[data-rating]"):
        rating_attr_node = film_node.select_one(selector)
        if rating_attr_node is None:
            continue
        raw_rating = (rating_attr_node.get("data-owner-rating") or rating_attr_node.get("data-rating") or "").strip()
        parsed = _parse_rating_value(raw_rating)
        if parsed is not None:
            return parsed

    rating_node = film_node.select_one("p.poster-viewingdata span.rating, span.rating")
    if rating_node is not None:
        rating_raw = re.sub(r"\s+", "", rating_node.get_text(strip=True))
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

    return None


def _extract_year(film_node: BeautifulSoup) -> Optional[int]:
    """Extract year from a film node (works for both grid and table views)."""
    # For diary table view, look for release date links
    year_link = film_node.select_one("td.td-released a, td.td-film-details small a")
    if year_link:
        year_text = year_link.get_text(strip=True)
        year_match = re.search(r"(\d{4})", year_text)
        if year_match:
            return int(year_match.group(1))
    
    # Legacy extraction for grid view
    for attr in ("data-film-release-year", "data-release-year", "data-year"):
        year_raw = film_node.get(attr)
        if year_raw and re.match(r"^\d{4}$", year_raw.strip()):
            return int(year_raw.strip())

    year_attr_node = film_node.select_one("[data-film-release-year], [data-release-year], [data-year]")
    if year_attr_node is not None:
        for attr in ("data-film-release-year", "data-release-year", "data-year"):
            year_raw = year_attr_node.get(attr)
            if year_raw and re.match(r"^\d{4}$", year_raw.strip()):
                return int(year_raw.strip())

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
    """Extract title from a film node (works for both grid and table views)."""
    # For diary table view, look for title in headline or link
    title_link = film_node.select_one("h3.headline-3 a, td.td-film-details h3 a, a.frame")
    if title_link:
        title = title_link.get_text(strip=True)
        if title:
            return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    
    # Legacy extraction for grid view
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


def _parse_films_page(soup: BeautifulSoup, debug: bool = False) -> Tuple[List[FilmRecord], int]:
    """Parse a films page and extract rated films."""
    # Try diary table view first (modern Letterboxd)
    films = soup.select("tr.diary-entry-row")
    is_diary_view = len(films) > 0
    
    # Fallback to grid view (older format)
    if not films:
        films = soup.select("li.poster-container, li.posteritem")
    if not films:
        films = soup.select("ul.poster-list > li")
    
    if debug:
        view_type = "diary table" if is_diary_view else "grid"
        print(f"[DEBUG] Found {len(films)} film containers ({view_type} view)")
    
    records: List[FilmRecord] = []
    skipped_no_title = 0
    skipped_no_rating = 0
    
    for film in films:
        title = _extract_title(film)
        if not title:
            skipped_no_title += 1
            continue

        year = _extract_year(film)
        rating = _extract_rating(film)
        if rating is None:
            skipped_no_rating += 1
            if debug:
                print(f"[DEBUG] Skipped film without rating: {title}")
            continue

        records.append(FilmRecord(title=title, year=year, rating=rating))
    
    if debug:
        print(f"[DEBUG] Parsed {len(records)} rated films, {skipped_no_title} without title, {skipped_no_rating} without rating")
    
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


def scrape_letterboxd_films(profile_or_films_url: str, output_csv: str = "user_data.csv", debug: bool = False) -> pd.DataFrame:
    """Scrape a Letterboxd user's films and ratings into a DataFrame and CSV."""
    films_url = _normalize_films_url(profile_or_films_url)

    session = _build_http_session()

    all_records: List[FilmRecord] = []
    page = 1

    while True:
        page_url = f"{films_url}/page/{page}/" if page > 1 else f"{films_url}/"
        try:
            response = session.get(page_url, timeout=20)
        except RequestException as exc:
            raise LetterboxdScrapeError(f"Network error while accessing Letterboxd: {exc}") from exc

        _validate_response(response, page_url)

        soup = BeautifulSoup(response.text, "html.parser")
        page_records, film_count = _parse_films_page(soup, debug=debug)
        if debug:
            print(f"[DEBUG] Page {page}: {film_count} film containers, {len(page_records)} with ratings")
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

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
    # Ensure trailing slash to avoid redirect issues
    if "/diary" in cleaned:
        return cleaned if cleaned.endswith("/") else f"{cleaned}/"
    elif "/films" in cleaned:
        new_url = cleaned.replace("/films", "/diary")
        return new_url if new_url.endswith("/") else f"{new_url}/"
    else:
        return f"{cleaned}/diary/"


def _build_http_session():
    """Build a robust session to bypass Letterboxd/Cloudflare protection."""
    common_headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://letterboxd.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
    }

    try:
        import cloudscraper
        # Use a more modern browser configuration for cloudscraper
        scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'linux', 'mobile': False},
            delay=10
        )
        scraper.trust_env = False
        scraper.headers.update(common_headers)
        return scraper
    except ImportError:
        # Fallback to regular session if cloudscraper not available
        session = requests.Session()
        session.trust_env = False
        session.headers.update(common_headers)
        return session


def _extract_rating(film_node: BeautifulSoup) -> Optional[float]:
    """Extract rating from a film node (works for both grid and table views)."""
    # Try extracting from common rating classes or structures
    rating_node = film_node.select_one(
        "span.rating[class*='rated-'], td.col-rating span.rating, div.rating-green span.rating, p.poster-viewingdata span.rating, span.rating"
    )
    if rating_node is not None:
        # Try class-based extraction from the matched node
        classes = rating_node.get("class", [])
        if not isinstance(classes, list):
            classes = [classes]
        for cls in classes:
            clean_cls = str(cls).strip("-")
            if clean_cls.startswith("rated-"):
                try:
                    rating_value = int(clean_cls.replace("rated-", ""))
                    return rating_value / 2.0
                except (ValueError, AttributeError):
                    pass
        
        # Try text-based extraction (stars)
        rating_raw = re.sub(r"\s+", "", rating_node.get_text(strip=True))
        mapped_rating = STAR_MAP.get(rating_raw)
        if mapped_rating is not None:
            return mapped_rating

    # Fallback: check all data-rating attributes on the node itself or its children
    for attr in ("data-owner-rating", "data-rating", "data-rating-value"):
        # Check node itself
        raw_rating = (film_node.get(attr) or "").strip()
        parsed = _parse_rating_value(raw_rating)
        if parsed is not None:
            return parsed
        # Check children
        nodes_with_attr = film_node.select(f"[{attr}]")
        for node in nodes_with_attr:
            raw_rating = (node.get(attr) or "").strip()
            parsed = _parse_rating_value(raw_rating)
            if parsed is not None:
                return parsed

    return None


def _extract_year(film_node: BeautifulSoup) -> Optional[int]:
    """Extract release year from the node."""
    # Try common year containers
    year_node = film_node.select_one(
        "td.td-released, td.col-releaseyear, span.release-year, span.metadata, span.releasedate, small.releasedate"
    )
    if year_node:
        try:
            match = re.search(r"(\d{4})", year_node.get_text())
            if match:
                return int(match.group(1))
        except (ValueError, TypeError):
            pass
    
    # Try data attributes on node itself or children
    for attr in ("data-film-release-year", "data-item-release-year", "data-release-year", "data-year"):
        # Check node itself
        year_raw = film_node.get(attr)
        if year_raw and re.match(r"^\d{4}$", str(year_raw).strip()):
            return int(str(year_raw).strip())
        # Check children
        nodes_with_attr = film_node.select(f"[{attr}]")
        for node in nodes_with_attr:
            year_raw = node.get(attr)
            if year_raw and re.match(r"^\d{4}$", str(year_raw).strip()):
                return int(str(year_raw).strip())

    return None


def _extract_title(film_node: BeautifulSoup) -> str:
    """Extract film title from the node."""
    title_link = film_node.select_one(
        "h3.headline-3 a, td.td-film-details h3 a, td.col-title a, a.frame, div.poster img"
    )
    if title_link:
        # Prefer alt text if it's an image, otherwise text
        if title_link.name == "img":
            title = title_link.get("alt", "").strip()
        else:
            title = title_link.get_text(strip=True)
        
        if title:
            return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    
    # Fallback to data-item-name
    name_node = film_node.select_one("[data-item-name]")
    if name_node:
        name = name_node.get("data-item-name", "").strip()
        if name:
            return re.sub(r"\s*\(\d{4}\)\s*$", "", name).strip()

    return ""


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
        if len(films) == 0:
            content_sample = soup.get_text()[:200]
            print(f"[DEBUG] Text Content Snippet: {content_sample}...")
    
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
    
    # Check for challenge or captcha pages
    if "just a moment..." in lower_text or "cloudflare" in lower_text and "challenge" in lower_text:
         raise LetterboxdScrapeError(
             f"Letterboxd access blocked by bot detection. Try running again in a few minutes or from a different IP."
         )


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

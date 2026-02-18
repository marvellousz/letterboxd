from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class TMDbClient:
    """Client for interacting with The Movie Database (TMDb) API v3."""

    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TMDB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TMDb API key is required. Pass it to the constructor or set TMDB_API_KEY environment variable."
            )
        self.session = requests.Session()

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params.copy() if params else {}
        params["api_key"] = self.api_key
        response = self.session.get(f"{self.BASE_URL}/{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def search_movie(self, title: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for a movie by title and optionally year."""
        params = {"query": title}
        if year:
            params["primary_release_year"] = year
        
        data = self._get("search/movie", params=params)
        return data.get("results", [])

    def get_movie_details(self, tmdb_id: int) -> Dict[str, Any]:
        """Get detailed information about a movie."""
        return self._get(f"movie/{tmdb_id}")

    def get_recommendations(self, tmdb_id: int) -> List[Dict[str, Any]]:
        """Get TMDb recommendations for a movie."""
        data = self._get(f"movie/{tmdb_id}/recommendations")
        return data.get("results", [])

    def get_similar(self, tmdb_id: int) -> List[Dict[str, Any]]:
        """Get similar movies from TMDb."""
        data = self._get(f"movie/{tmdb_id}/similar")
        return data.get("results", [])

    def get_poster_url(self, poster_path: Optional[str], size: str = "w500") -> Optional[str]:
        """Construct a full poster URL."""
        if not poster_path:
            return None
        return f"https://image.tmdb.org/t/p/{size}{poster_path}"

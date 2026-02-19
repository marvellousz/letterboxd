from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from tmdb_client import TMDbClient

MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-latest.zip"


@dataclass
class ModelArtifacts:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    user_history: pd.DataFrame
    candidates: pd.DataFrame
    tmdb_ids: Dict[int, int]  # Mapping from MovieLens ID -> TMDb ID
    seen_titles: List[str]  # All titles the user has reviewed


def _download_movielens(data_dir: str = "data") -> Path:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    extract_dir = data_path / "ml-latest"

    if (extract_dir / "ratings.csv").exists() and (extract_dir / "movies.csv").exists():
        return extract_dir

    zip_path = data_path / "ml-latest.zip"
    print(f"  Downloading full MovieLens dataset (~300MB) from {MOVIELENS_URL}...")
    response = requests.get(MOVIELENS_URL, timeout=120, stream=True)
    response.raise_for_status()
    
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("  Extracting dataset...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_path)

    return extract_dir


def _calculate_percentile(rating: float, all_ratings: list) -> float:
    """Calculate percentile rank for a rating (Criticker-style).
    
    Returns 0-99 percentile based on how many ratings are lower.
    """
    if len(all_ratings) < 5:  # Need minimum variation
        return 50.0
    
    num_lower = sum(1 for r in all_ratings if r < rating)
    num_matching = sum(1 for r in all_ratings if r == rating)
    total = len(all_ratings)
    
    percentile = int(((num_lower + (num_matching / 2.0)) / total) * 100)
    return float(percentile)


def _ratings_to_percentiles(ratings_df: pd.DataFrame, user_col: str = "userId") -> pd.DataFrame:
    """Convert ratings to percentiles for each user using vectorized operations."""
    # Filter out users with < 10 ratings to improve quality and performance
    user_counts = ratings_df[user_col].value_counts()
    valid_users = user_counts[user_counts >= 10].index
    filtered_df = ratings_df[ratings_df[user_col].isin(valid_users)].copy()

    # Calculate percentiles (0-99) for each user's ratings
    filtered_df["percentile"] = (
        filtered_df.groupby(user_col)["rating"]
        .rank(pct=True, method="average")
        .mul(100)
        .astype(int)
    ).astype(float)
    
    return filtered_df


def _calculate_tci(user_percentiles: dict, ml_user_percentiles: dict) -> tuple:
    """Calculate Taste Compatibility Index (average percentile difference).
    
    Returns (TCI, num_common) where lower TCI = more similar taste.
    """
    common_movies = set(user_percentiles.keys()) & set(ml_user_percentiles.keys())
    
    if len(common_movies) < 3:  # Need minimum overlap
        return (float('inf'), 0)
    
    differences = [abs(user_percentiles[mid] - ml_user_percentiles[mid]) for mid in common_movies]
    tci = sum(differences) / len(differences)
    
    return (tci, len(common_movies))


def _normalize_title(title: str) -> str:
    return "".join(ch.lower() for ch in title if ch.isalnum() or ch.isspace()).strip()


def _match_user_to_movielens(user_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    movies = movies_df.copy()
    movies["title_no_year"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
    movies["year"] = (
        movies["title"].str.extract(r"\((\d{4})\)$")[0].astype("float").astype("Int64")
    )
    movies["norm_title"] = movies["title_no_year"].map(_normalize_title)

    user = user_df.copy()
    user["norm_title"] = user["title"].map(_normalize_title)

    matched = user.merge(
        movies[["movieId", "title", "norm_title", "year", "genres"]],
        on=["norm_title", "year"],
        how="left",
        suffixes=("_user", "_ml"),
    )

    fallback = matched[matched["movieId"].isna()].drop(columns=["movieId", "title_ml", "genres"])
    if not fallback.empty:
        fallback = fallback.merge(
            movies[["movieId", "title", "norm_title", "genres"]],
            on="norm_title",
            how="left",
            suffixes=("_user", "_ml"),
        )
        fallback = fallback.sort_values(by="movieId").drop_duplicates(subset=["title_user", "year"])

        matched_non_null = matched[matched["movieId"].notna()]
        matched = pd.concat([matched_non_null, fallback], ignore_index=True)

    matched = matched.dropna(subset=["movieId"]).copy()
    matched["movieId"] = matched["movieId"].astype(int)
    return matched


def _get_tmdb_id_for_ml(movie_id: int, links_df: pd.DataFrame) -> Optional[int]:
    """Get TMDb ID for a MovieLens ID."""
    match = links_df[links_df["movieId"] == movie_id]
    if not match.empty and not pd.isna(match.iloc[0]["tmdbId"]):
        return int(match.iloc[0]["tmdbId"])
    return None


def train_and_recommend(
    user_data_path: str = "user_data.csv",
    top_k: int = 50,
    data_dir: str = "data",
    tmdb_client: Optional[TMDbClient] = None,
) -> ModelArtifacts:
    """Generate recommendations using Criticker-style percentile matching."""
    user_df = pd.read_csv(user_data_path)
    if user_df.empty:
        raise ValueError("user_data.csv is empty. Scrape user data first.")

    ml_dir = _download_movielens(data_dir=data_dir)
    ratings = pd.read_csv(ml_dir / "ratings.csv")
    movies = pd.read_csv(ml_dir / "movies.csv")
    links = pd.read_csv(ml_dir / "links.csv")

    user_history = _match_user_to_movielens(user_df, movies)
    if user_history.empty:
        raise ValueError(
            "Could not match Letterboxd films to MovieLens catalog. Try a user with more mainstream films."
        )
    
    # Convert user's ratings to percentiles
    user_ratings_list = user_history["rating"].tolist()
    if len(set(user_ratings_list)) < 5:
        raise ValueError("Need at least 5 distinct ratings for percentile calculation.")
    
    user_percentiles = {}
    tmdb_mapping: Dict[int, int] = {}
    for _, row in user_history.iterrows():
        movie_id = int(row["movieId"])
        percentile = _calculate_percentile(row["rating"], user_ratings_list)
        user_percentiles[movie_id] = percentile
        
        tmdb_id = _get_tmdb_id_for_ml(movie_id, links)
        if tmdb_id:
            tmdb_mapping[movie_id] = tmdb_id
    
    # Convert MovieLens ratings to percentiles (sample for performance)
    print("  Computing percentiles for MovieLens users...")
    ml_percentiles_df = _ratings_to_percentiles(ratings, user_col="userId")
    
    # Calculate TCI with MovieLens users in bulk
    print("  Finding taste-compatible users (vectorized)...")
    # Only consider users who have rated AT LEAST one movie from our history
    ml_relevant = ml_percentiles_df[ml_percentiles_df["movieId"].isin(user_percentiles.keys())].copy()
    
    # Map user's own percentiles to this dataframe
    ml_relevant["user_percentile"] = ml_relevant["movieId"].map(user_percentiles)
    
    # Absolute difference
    ml_relevant["diff"] = (ml_relevant["percentile"] - ml_relevant["user_percentile"]).abs()
    
    # Group by user to get TCI and count
    user_tci = ml_relevant.groupby("userId").agg({
        "diff": "mean",
        "movieId": "count"
    }).rename(columns={"diff": "tci", "movieId": "num_common"})
    
    # Filter by minimum overlap
    min_common = max(5, int(len(user_percentiles) * 0.10))
    tci_filtered = user_tci[user_tci["num_common"] >= min_common].sort_values("tci")
    
    if tci_filtered.empty:
        raise ValueError("No taste-compatible users found. Try rating more movies or common films.")
        
    top_compatible_users = tci_filtered.head(500).index.tolist()
    print(f"  Found {len(top_compatible_users)} compatible users (best TCI: {tci_filtered.iloc[0]['tci']:.1f})")
    
    # Generate PSI (Probable Score Indicator) using group operations
    print("  Calculating probable scores...")
    seen_movie_ids = set(user_percentiles.keys())
    
    # Only look at data from our top compatible users
    relevant_ratings = ml_percentiles_df[
        (ml_percentiles_df["userId"].isin(top_compatible_users)) & 
        (~ml_percentiles_df["movieId"].isin(seen_movie_ids))
    ]
    
    # Average percentile for each movie
    movie_stats = relevant_ratings.groupby("movieId").agg({
        "percentile": ["mean", "count"]
    })
    movie_stats.columns = ["avg_percentile", "count"]
    
    # Filter for movies rated by enough compatible users
    min_ratings_from_compatible = 3
    top_candidates = movie_stats[movie_stats["count"] >= min_ratings_from_compatible].copy()
    
    # Convert back to user rating scale
    sorted_user_ratings = sorted(user_ratings_list)
    def percentile_to_rating(p):
        idx = int((p / 100.0) * len(sorted_user_ratings))
        return sorted_user_ratings[min(idx, len(sorted_user_ratings) - 1)]
    
    top_candidates["cf_score"] = top_candidates["avg_percentile"].apply(percentile_to_rating)
    
    pred_df = top_candidates.reset_index().sort_values("cf_score", ascending=False)

    # TMDb Expansion: Use TMDb recommendations for top-rated films
    if tmdb_client is not None:
        print("  Expanding candidates using TMDb...")
        favorites = user_history.sort_values(by="rating", ascending=False).head(5)
        tmdb_candidates = []
        for _, fav in favorites.iterrows():
            ml_id = int(fav["movieId"])
            tmdb_id = tmdb_mapping.get(ml_id)
            if tmdb_id:
                try:
                    recs = tmdb_client.get_recommendations(tmdb_id)
                    for rec in recs:
                        # Try to find this TMDb movie in MovieLens links
                        ml_match = links[links["tmdbId"] == rec["id"]]
                        if not ml_match.empty:
                            new_ml_id = int(ml_match.iloc[0]["movieId"])
                            if new_ml_id not in seen_movie_ids:
                                tmdb_candidates.append({
                                    "movieId": new_ml_id,
                                    "cf_score": fav["rating"] * 0.9, # Slight penalty for proximity
                                    "avg_percentile": 75.0 # Default decent percentile
                                })
                                tmdb_mapping[new_ml_id] = rec["id"]
                except Exception as e:
                    print(f"    Warning: TMDb lookup failed for {fav['title']}: {e}")
        
        if tmdb_candidates:
            tmdb_df = pd.DataFrame(tmdb_candidates).drop_duplicates(subset=["movieId"])
            pred_df = pd.concat([pred_df, tmdb_df], ignore_index=True).sort_values(
                by="cf_score", ascending=False
            )

    candidates = pred_df.head(top_k).merge(movies, on="movieId", how="left")
    candidates = candidates.sort_values(by="cf_score", ascending=False)
    candidates["title_clean"] = candidates["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)

    return ModelArtifacts(
        movies=movies, 
        ratings=ratings, 
        user_history=user_history, 
        candidates=candidates,
        tmdb_ids=tmdb_mapping,
        seen_titles=user_df["title"].tolist()
    )


if __name__ == "__main__":
    artifacts = train_and_recommend()
    print(artifacts.candidates[["title", "cf_score"]].head(10).to_string(index=False))

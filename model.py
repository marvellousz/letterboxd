from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


@dataclass
class ModelArtifacts:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    user_history: pd.DataFrame
    candidates: pd.DataFrame


def _download_movielens(data_dir: str = "data") -> Path:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    extract_dir = data_path / "ml-latest-small"

    if (extract_dir / "ratings.csv").exists() and (extract_dir / "movies.csv").exists():
        return extract_dir

    zip_path = data_path / "ml-latest-small.zip"
    response = requests.get(MOVIELENS_URL, timeout=30)
    response.raise_for_status()
    zip_path.write_bytes(response.content)

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
    """Convert ratings to percentiles for each user (Criticker-style)."""
    result_rows = []
    
    for user_id in ratings_df[user_col].unique():
        user_ratings = ratings_df[ratings_df[user_col] == user_id].copy()
        all_user_ratings = user_ratings["rating"].tolist()
        
        if len(set(all_user_ratings)) < 5:  # Skip users with < 5 distinct ratings
            continue
        
        for _, row in user_ratings.iterrows():
            percentile = _calculate_percentile(row["rating"], all_user_ratings)
            result_rows.append({
                user_col: user_id,
                "movieId": row["movieId"],
                "rating": row["rating"],
                "percentile": percentile
            })
    
    return pd.DataFrame(result_rows)


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


def train_and_recommend(
    user_data_path: str = "user_data.csv",
    top_k: int = 50,
    data_dir: str = "data",
) -> ModelArtifacts:
    """Generate recommendations using Criticker-style percentile matching."""
    user_df = pd.read_csv(user_data_path)
    if user_df.empty:
        raise ValueError("user_data.csv is empty. Scrape user data first.")

    ml_dir = _download_movielens(data_dir=data_dir)
    ratings = pd.read_csv(ml_dir / "ratings.csv")
    movies = pd.read_csv(ml_dir / "movies.csv")

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
    for _, row in user_history.iterrows():
        movie_id = int(row["movieId"])
        percentile = _calculate_percentile(row["rating"], user_ratings_list)
        user_percentiles[movie_id] = percentile
    
    # Convert MovieLens ratings to percentiles (sample for performance)
    print("  Computing percentiles for MovieLens users...")
    ml_percentiles_df = _ratings_to_percentiles(ratings, user_col="userId")
    
    # Calculate TCI with each MovieLens user
    print("  Finding taste-compatible users...")
    tci_scores = []
    min_common = max(3, int(len(user_percentiles) * 0.15))  # At least 15% overlap
    
    for ml_user_id in ml_percentiles_df["userId"].unique():
        ml_user_data = ml_percentiles_df[ml_percentiles_df["userId"] == ml_user_id]
        ml_user_percentiles = dict(zip(ml_user_data["movieId"], ml_user_data["percentile"]))
        
        tci, num_common = _calculate_tci(user_percentiles, ml_user_percentiles)
        
        if num_common >= min_common and tci != float('inf'):
            tci_scores.append((ml_user_id, tci, num_common))
    
    if len(tci_scores) < 10:
        raise ValueError("Not enough taste-compatible users found. Try rating more films.")
    
    # Get top 200 most compatible users (lowest TCI)
    tci_scores.sort(key=lambda x: (x[1], -x[2]))  # Sort by TCI (lower is better), then by overlap
    top_compatible_users = [uid for uid, _, _ in tci_scores[:200]]
    
    print(f"  Found {len(top_compatible_users)} compatible users (best TCI: {tci_scores[0][1]:.1f})")
    
    # Generate PSI (Probable Score Indicator) for unseen movies
    seen_movie_ids = set(user_percentiles.keys())
    all_movie_ids = set(movies["movieId"].astype(int).tolist())
    candidate_ids = all_movie_ids - seen_movie_ids
    
    predictions = []
    compatible_user_data = ml_percentiles_df[ml_percentiles_df["userId"].isin(top_compatible_users)]
    
    for movie_id in candidate_ids:
        # Get percentiles from compatible users who rated this movie
        movie_percentiles = compatible_user_data[compatible_user_data["movieId"] == movie_id]["percentile"].tolist()
        
        if len(movie_percentiles) < 3:  # Need at least 3 ratings
            continue
        
        # Average percentile from compatible users
        avg_percentile = sum(movie_percentiles) / len(movie_percentiles)
        
        # Convert percentile back to user's rating scale (PSI)
        # Find rating at this percentile in user's distribution
        sorted_ratings = sorted(user_ratings_list)
        idx = int((avg_percentile / 100.0) * len(sorted_ratings))
        idx = min(idx, len(sorted_ratings) - 1)
        predicted_rating = sorted_ratings[idx]
        
        predictions.append((movie_id, predicted_rating, avg_percentile))
    
    pred_df = pd.DataFrame(predictions, columns=["movieId", "cf_score", "avg_percentile"]).sort_values(
        by="cf_score", ascending=False
    )

    candidates = pred_df.head(top_k).merge(movies, on="movieId", how="left")
    candidates["title_clean"] = candidates["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)

    return ModelArtifacts(movies=movies, ratings=ratings, user_history=user_history, candidates=candidates)


if __name__ == "__main__":
    artifacts = train_and_recommend()
    print(artifacts.candidates[["title", "cf_score"]].head(10).to_string(index=False))

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from surprise import Dataset, Reader, SVD

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
    """Train an SVD collaborative filtering model and generate candidate recommendations."""
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

    reader = Reader(rating_scale=(0.5, 5.0))
    train_rows = ratings[["userId", "movieId", "rating"]].copy()
    personal_rows = user_history[["movieId", "rating"]].copy()
    personal_rows["userId"] = "letterboxd_user"

    full_rows = pd.concat([train_rows, personal_rows[["userId", "movieId", "rating"]]], ignore_index=True)
    dataset = Dataset.load_from_df(full_rows[["userId", "movieId", "rating"]], reader)
    trainset = dataset.build_full_trainset()

    model = SVD(n_factors=100, n_epochs=25, lr_all=0.005, reg_all=0.02, random_state=42)
    model.fit(trainset)

    seen_movie_ids = set(user_history["movieId"].astype(int).tolist())
    all_movie_ids = set(movies["movieId"].astype(int).tolist())
    candidate_ids = sorted(all_movie_ids - seen_movie_ids)

    predictions = [
        (movie_id, model.predict("letterboxd_user", movie_id).est)
        for movie_id in candidate_ids
    ]
    pred_df = pd.DataFrame(predictions, columns=["movieId", "cf_score"]).sort_values(
        by="cf_score", ascending=False
    )

    candidates = pred_df.head(top_k).merge(movies, on="movieId", how="left")
    candidates["title_clean"] = candidates["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)

    return ModelArtifacts(movies=movies, ratings=ratings, user_history=user_history, candidates=candidates)


if __name__ == "__main__":
    artifacts = train_and_recommend()
    print(artifacts.candidates[["title", "cf_score"]].head(10).to_string(index=False))

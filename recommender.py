from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from tmdb_client import TMDbClient


@dataclass
class Recommendation:
    title: str
    year: str
    score: float
    cf_score: float
    vibe_similarity: float
    why: str
    overview: Optional[str] = None
    poster_url: Optional[str] = None


def _safe_norm(vector: np.ndarray) -> float:
    norm = float(np.linalg.norm(vector))
    return norm if norm > 1e-12 else 1e-12


def _cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector_norm = _safe_norm(vector)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    matrix_norm[matrix_norm < 1e-12] = 1e-12
    return (matrix @ vector) / (matrix_norm * vector_norm)


def _build_overview_text(df: pd.DataFrame) -> pd.Series:
    if "overview" in df.columns and df["overview"].notna().any():
        return df["overview"].fillna("")
    return (
        "Movie: "
        + df["title_clean"].fillna("")
        + ". Genres: "
        + df.get("genres", pd.Series([""] * len(df))).fillna("")
        + "."
    )


def refine_with_vibe(
    candidates_df: pd.DataFrame,
    user_history_df: pd.DataFrame,
    tmdb_ids: Dict[int, int],
    tmdb_client: Optional[TMDbClient] = None,
    top_n: int = 5,
    seen_titles: Optional[List[str]] = None,
) -> List[Recommendation]:
    """Refine candidate list by comparing semantic vibe to highly-rated films."""
    candidates = candidates_df.copy().reset_index(drop=True)
    favorites = user_history_df.sort_values(by="rating", ascending=False).head(10)
    
    # 0. Robust filtering of already seen movies by title
    def normalize(t):
        return "".join(ch.lower() for ch in str(t) if ch.isalnum()).strip()

    if seen_titles:
        normalized_seen = {normalize(t) for t in seen_titles}
        # Also include titles from user_history_df just in case
        if "title_user" in user_history_df.columns:
            normalized_seen.update({normalize(t) for t in user_history_df["title_user"]})
        
        candidates["norm_title_for_filter"] = candidates["title_clean"].apply(normalize)
        candidates = candidates[~candidates["norm_title_for_filter"].isin(normalized_seen)]
        candidates = candidates.drop(columns=["norm_title_for_filter"])

    if candidates.empty:
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")

    candidate_texts = _build_overview_text(candidates).tolist()
    candidate_embeddings = model.encode(candidate_texts, convert_to_numpy=True)

    # Build user profile from their favorite films
    if not favorites.empty:
        # Get genres from favorites to personalize recommendations
        favorite_genres = favorites.get("genres", pd.Series([""] * len(favorites))).fillna("")
        favorite_titles = favorites["title_user"].fillna("").tolist()
        
        # Create prompts from actual user favorites (no hardcoded bias)
        favorite_prompts = [f"Movie: {title}. Genres: {genre}" 
                          for title, genre in zip(favorite_titles, favorite_genres)]
        favorite_embeddings = model.encode(favorite_prompts, convert_to_numpy=True)
        user_centroid = favorite_embeddings.mean(axis=0)
    else:
        # Fallback to generic if no favorites
        user_centroid = model.encode(["highly rated movies"], convert_to_numpy=True)[0]

    # Use only user preferences for vibe similarity (no hardcoded theme)
    vibe_similarity = _cosine_similarity(candidate_embeddings, user_centroid)

    cf = candidates["cf_score"].to_numpy(dtype=float)
    cf_norm = (cf - cf.min()) / (cf.max() - cf.min() + 1e-9)
    vibe_norm = (vibe_similarity - vibe_similarity.min()) / (vibe_similarity.max() - vibe_similarity.min() + 1e-9)

    # Give more weight to vibe similarity for better personalization
    final_score = 0.40 * cf_norm + 0.60 * vibe_norm

    candidates["vibe_similarity"] = vibe_similarity
    candidates["final_score"] = final_score
    ranked = candidates.sort_values(by="final_score", ascending=False).head(top_n)

    recommendations: List[Recommendation] = []
    for _, row in ranked.iterrows():
        title_full = str(row["title"])
        year = ""
        if "(" in title_full and ")" in title_full:
            year = title_full.rsplit("(", 1)[-1].replace(")", "")

        why = (
            f"Strong collaborative-fit score ({row['cf_score']:.2f}) and aligned vibe similarity "
            f"({row['vibe_similarity']:.2f}) with your top-rated films."
        )

        overview = None
        poster_url = None
        
        if tmdb_client and row["movieId"] in tmdb_ids:
            tmdb_id = tmdb_ids[row["movieId"]]
            try:
                details = tmdb_client.get_movie_details(tmdb_id)
                overview = details.get("overview")
                poster_url = tmdb_client.get_poster_url(details.get("poster_path"))
            except Exception as e:
                print(f"    Warning: Could not fetch TMDb details for {title_full}: {e}")

        recommendations.append(
            Recommendation(
                title=row.get("title_clean", title_full),
                year=year,
                score=float(row["final_score"]),
                cf_score=float(row["cf_score"]),
                vibe_similarity=float(row["vibe_similarity"]),
                why=why,
                overview=overview,
                poster_url=poster_url,
            )
        )

    return recommendations

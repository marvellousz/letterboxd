from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class Recommendation:
    title: str
    year: str
    score: float
    cf_score: float
    vibe_similarity: float
    why: str


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
    top_n: int = 5,
) -> List[Recommendation]:
    """Refine candidate list by comparing semantic vibe to highly-rated films."""
    candidates = candidates_df.copy().reset_index(drop=True)

    favorites = user_history_df.sort_values(by="rating", ascending=False).head(10)
    
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

        recommendations.append(
            Recommendation(
                title=row.get("title_clean", title_full),
                year=year,
                score=float(row["final_score"]),
                cf_score=float(row["cf_score"]),
                vibe_similarity=float(row["vibe_similarity"]),
                why=why,
            )
        )

    return recommendations

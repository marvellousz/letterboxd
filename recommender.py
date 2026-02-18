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
    favorite_titles = favorites["title_user"].fillna("").tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    candidate_texts = _build_overview_text(candidates).tolist()
    candidate_embeddings = model.encode(candidate_texts, convert_to_numpy=True)

    if favorite_titles:
        favorite_prompts = [f"Movie with philosophical depth or cyberpunk/noir mood: {title}" for title in favorite_titles]
        favorite_embeddings = model.encode(favorite_prompts, convert_to_numpy=True)
        user_centroid = favorite_embeddings.mean(axis=0)
    else:
        user_centroid = model.encode(
            ["philosophical depth, cyberpunk cityscapes, noir atmosphere, existential themes"],
            convert_to_numpy=True,
        )[0]

    vibe_query = model.encode(
        ["philosophical depth, cyberpunk/noir tone, existential conflict, futuristic melancholy"],
        convert_to_numpy=True,
    )[0]

    sim_user = _cosine_similarity(candidate_embeddings, user_centroid)
    sim_query = _cosine_similarity(candidate_embeddings, vibe_query)
    vibe_similarity = 0.65 * sim_user + 0.35 * sim_query

    cf = candidates["cf_score"].to_numpy(dtype=float)
    cf_norm = (cf - cf.min()) / (cf.max() - cf.min() + 1e-9)
    vibe_norm = (vibe_similarity - vibe_similarity.min()) / (vibe_similarity.max() - vibe_similarity.min() + 1e-9)

    final_score = 0.55 * cf_norm + 0.45 * vibe_norm

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

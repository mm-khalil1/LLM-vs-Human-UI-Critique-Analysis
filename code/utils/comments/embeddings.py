from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

DEFAULT_EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 128


def prepare_match_text(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_default_embedding_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    *,
    device: Optional[str] = None,
    token: Optional[str] = None,
) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device or get_default_embedding_device(), token=token)

def build_embeddings_df(
    items_df: pd.DataFrame,
    *,
    id_col: str,
    text_col: str,
    model: SentenceTransformer,
    embedding_col: str = "embedding",
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    show_progress_bar: bool = True,
) -> pd.DataFrame:
    embedding_input_df = items_df[[id_col, text_col]].drop_duplicates(subset=[id_col]).copy()
    embedding_input_df[text_col] = embedding_input_df[text_col].apply(prepare_match_text)

    embeddings = model.encode(
        embedding_input_df[text_col].tolist(),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return pd.DataFrame(
        {
            id_col: embedding_input_df[id_col].tolist(),
            embedding_col: list(embeddings),
        }
    )

def build_comment_embeddings(
    comments_df: pd.DataFrame,
    *,
    id_col: str,
    text_col: str,
    model: SentenceTransformer,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    show_progress_bar: bool = True,
) -> dict[str, np.ndarray]:
    embeddings_df = build_embeddings_df(
        comments_df,
        id_col=id_col,
        text_col=text_col,
        model=model,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    return dict(zip(embeddings_df[id_col], embeddings_df["embedding"]))

def add_pair_cosine_similarity(
    pairs_df: pd.DataFrame,
    *,
    left_id_col: str,
    right_id_col: str,
    left_embeddings: dict,
    right_embeddings: dict,
    output_col: str = "cosine_similarity",
) -> pd.DataFrame:
    scored_df = pairs_df.copy()
    left_vectors = np.vstack(scored_df[left_id_col].map(left_embeddings).to_numpy())
    right_vectors = np.vstack(scored_df[right_id_col].map(right_embeddings).to_numpy())
    scored_df[output_col] = np.einsum("ij,ij->i", left_vectors, right_vectors)
    return scored_df

def cosine_similarity(vec_a: object, vec_b: object) -> float:
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def score_standardized_pairs(
    pairs_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    *,
    embedding_model: SentenceTransformer,
    comment_id_col: str = "comment_id",
    match_text_col: str = "match_text",
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    show_progress_bar: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    embeddings_df = build_embeddings_df(
        comments_df[[comment_id_col, match_text_col]].copy(),
        id_col=comment_id_col,
        text_col=match_text_col,
        model=embedding_model,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )

    scored_df = pairs_df.merge(
        embeddings_df.rename(columns={comment_id_col: "left_comment_id", "embedding": "left_embedding"}),
        on="left_comment_id",
        how="left",
    ).merge(
        embeddings_df.rename(columns={comment_id_col: "right_comment_id", "embedding": "right_embedding"}),
        on="right_comment_id",
        how="left",
    )

    scored_df["cosine_similarity"] = scored_df.apply(
        lambda row: cosine_similarity(row["left_embedding"], row["right_embedding"]),
        axis=1,
    )
    return scored_df, embeddings_df

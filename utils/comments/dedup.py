"""Semantic deduplication of expert comments within a task.

Candidate pairs are scored by embedding cosine similarity, judged by LLMs (see judge.DUPLICATE_REVIEW),
optionally reviewed by hand, and the resulting keep_A/keep_B decisions are resolved into comments to drop.
Pairs carry left_/right_ comment columns and a `final_decision` column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.screen_display import show_screen

DECISION_COMMANDS = {"a": "keep_A", "b": "keep_B", "both": "keep_both"}
INFO_COLUMN_PREFIXES = ("relationship", "recommended_action", "confidence", "final_decision", "decision_source", "bbox_iou", "component")


def cosine_similarity(
    pairs_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    *,
    model_name: str = "all-mpnet-base-v2",
    batch_size: int = 128,
) -> pd.Series:
    """Cosine similarity of each pair's observed_issue embeddings (comments_df has comment_id, observed_issue)."""
    from sentence_transformers import SentenceTransformer

    comments_df = comments_df.drop_duplicates("comment_id")
    texts = comments_df["observed_issue"].map(lambda text: "" if pd.isna(text) else str(text).strip())
    embeddings = SentenceTransformer(model_name).encode(
        texts.tolist(), batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
    )
    vectors = dict(zip(comments_df["comment_id"], embeddings.astype(float)))
    return pd.Series(
        [
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            for a, b in zip(pairs_df["left_comment_id"].map(vectors), pairs_df["right_comment_id"].map(vectors))
        ],
        index=pairs_df.index,
    )


def load_decisions(path: Path) -> dict[str, str]:
    """Manual decisions saved by review_pairs: {custom_id: keep_A | keep_B | keep_both}."""
    if not Path(path).exists():
        return {}
    return pd.read_csv(path).set_index("custom_id")["final_decision"].to_dict()


def review_pairs(pairs_df: pd.DataFrame, decisions_path: Path, *, width: int = 300) -> None:
    """Step through pairs that have no saved decision yet; every decision is written to decisions_path at once."""
    decisions = load_decisions(decisions_path)
    pairs_df = pairs_df[~pairs_df["custom_id"].isin(decisions)]
    commands = "a=keep_A, b=keep_B, both=keep_both, n/Enter=skip, p=previous, q=quit"
    i = 0
    while 0 <= i < len(pairs_df):
        row = pairs_df.iloc[i]
        print("\n" + "=" * 110)
        print(f"{i + 1}/{len(pairs_df)} | {row['custom_id']} | {row['screen_task_id']}: {row['task']}")
        for col in row.index:
            if col.startswith(INFO_COLUMN_PREFIXES):
                print(f"{col:<40}: {row[col]}")
        bboxes = {label: row.get(f"{side}_bounding_box") for side, label in (("left", "A"), ("right", "B"))}
        bboxes = {label: bbox for label, bbox in bboxes.items() if isinstance(bbox, dict)}
        show_screen(screen_id=row["screen_id"], bboxes=list(bboxes.values()), labels=list(bboxes), width=width)
        print("-" * 110)
        print("A:", row["left_observed_issue"])
        print("B:", row["right_observed_issue"])
        print("Saved decision:", decisions.get(row["custom_id"]))

        command = input(f"{commands}: ").strip().lower()
        if command == "q":
            break
        if command == "p":
            i = max(0, i - 1)
            continue
        if command in DECISION_COMMANDS:
            decisions[row["custom_id"]] = DECISION_COMMANDS[command]
            pd.Series(decisions, name="final_decision").rename_axis("custom_id").sort_index().to_csv(decisions_path)
        elif command not in {"", "n"}:
            print("Unknown command.")
            continue
        i += 1


def merge_components(decisions_df: pd.DataFrame) -> tuple[set[str], pd.DataFrame]:
    """Resolve keep_A/keep_B decisions into the comment ids to drop.

    Comments linked by keep_A/keep_B decisions form a component. A component may keep several comments (its
    survivors were judged different issues), but if every comment in it is dropped the decisions contradict
    each other (e.g. A drops B and B drops A). Returns the dropped ids and the pairs of such components.
    """
    left_ids = decisions_df["left_comment_id"].astype(str)
    right_ids = decisions_df["right_comment_id"].astype(str)
    is_merge = decisions_df["final_decision"].isin(["keep_A", "keep_B"])
    keeps_left = decisions_df["final_decision"].eq("keep_A")
    drop_ids = set(right_ids.where(keeps_left, left_ids)[is_merge])

    parent = {}

    def root(comment_id: str) -> str:
        while parent.setdefault(comment_id, comment_id) != comment_id:
            comment_id = parent[comment_id]
        return comment_id

    for left_id, right_id in zip(left_ids[is_merge], right_ids[is_merge]):
        parent[root(right_id)] = root(left_id)

    component_of = {comment_id: root(comment_id) for comment_id in list(parent)}
    components = {}
    for comment_id, component_id in component_of.items():
        components.setdefault(component_id, set()).add(comment_id)
    problem_components = {component_id for component_id, ids in components.items() if ids <= drop_ids}

    left_component = left_ids.map(component_of)
    in_problem = left_component.isin(problem_components) & left_component.eq(right_ids.map(component_of))
    problem_rows_df = decisions_df[in_problem].assign(component_id=left_component[in_problem])
    return drop_ids, problem_rows_df.sort_values("component_id", kind="stable")

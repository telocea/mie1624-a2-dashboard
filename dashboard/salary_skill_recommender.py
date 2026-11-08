"""Rank encoded skill/technology additions by predicted salary-bucket shift.

This module expects the fitted pipeline and one respondent's encoded feature
row from the notebook. It does not claim causal salary effects.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


# One-hot groups that can reasonably represent a skill, technology, or tool
# someone might adopt. Extend this tuple if the model includes another such
# family. Feature names must be encoded model input columns.
DEFAULT_CANDIDATE_PREFIXES = (
    "LanguagesHaveWorkedWith__",
    "LanguagesUsedAI__",
    "CodeTestHaveWorkedWith__",
    "DatabasesHaveWorkedWith__",
    "CloudPlatHaveWorkedWith__",
    "WebFrameworkHaveWorkedWith__",
    "DevIDEHaveWorkedWith__",
    "CommPlatHaveWorkedWith__",
    "LLMHaveWorkedWith__",
    "AICodeAgentHaveWorkedWith__",
    "AISelect__",
    "KnowMgmtHaveWorkedWith__",
    "BuildTool__",
    "BuildToolAI__",
)


def recommend_skills(
    model,
    respondent: pd.DataFrame,
    *,
    classes: Iterable | None = None,
    candidate_prefixes: Iterable[str] = DEFAULT_CANDIDATE_PREFIXES,
    top_n: int = 3,
    tie_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Rank one-feature additions by change in expected ordinal bucket rank.

    The improvement score is E[bucket rank | candidate] - E[bucket rank | base],
    with ordered model classes assigned ranks 0..K-1. This avoids pretending
    that open-ended or unequal-width salary buckets have exact dollar values.
    Candidate actions set one currently-zero encoded dummy to one. Only model
    input columns in the specified feature families are considered.

    `model` is the fitted sklearn Pipeline or fitted estimator with
    `predict_proba`. `respondent` must be a one-row DataFrame in model input
    column order.
    """
    if not isinstance(respondent, pd.DataFrame) or len(respondent) != 1:
        raise ValueError("respondent must be a one-row pandas DataFrame")
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must be a fitted estimator or pipeline with predict_proba")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    p0 = np.asarray(model.predict_proba(respondent), dtype=float)
    if p0.shape != (1, p0.shape[1]) or p0.shape[1] < 2:
        raise ValueError(f"Unexpected probability shape: {p0.shape}")
    cls = np.asarray(list(classes)) if classes is not None else None
    if cls is None:
        cls = getattr(model, "classes_", None)
        if cls is None and hasattr(model, "steps"):
            cls = getattr(model.steps[-1][1], "classes_", None)
        if cls is None:
            raise ValueError("Pass ordered `classes` or use a fitted model exposing classes_")
        cls = np.asarray(cls)
    if len(cls) != p0.shape[1]:
        raise ValueError("The number of classes does not match predict_proba output")

    # The supplied classes are assumed to be in low-to-high salary order.
    rank = np.arange(len(cls), dtype=float)
    baseline_expected_rank = float(p0[0] @ rank)
    prefixes = tuple(candidate_prefixes)
    cols = [c for c in respondent.columns if c.startswith(prefixes)]
    candidates = [c for c in cols if pd.to_numeric(respondent.iloc[0][c], errors="coerce") == 0]
    if not candidates:
        raise ValueError("No currently-zero candidate feature columns found in the respondent row")

    rows = []
    for feature in candidates:
        scenario = respondent.copy()
        scenario.loc[scenario.index[0], feature] = 1.0
        p = np.asarray(model.predict_proba(scenario), dtype=float)[0]
        expected_rank = float(p @ rank)
        rows.append({
            "feature": feature,
            "skill_or_tool": feature.split("__", 1)[1],
            "baseline_expected_bucket_rank": baseline_expected_rank,
            "scenario_expected_bucket_rank": expected_rank,
            "expected_rank_improvement": expected_rank - baseline_expected_rank,
            "baseline_probabilities": p0[0].tolist(),
            "scenario_probabilities": p.tolist(),
        })
    result = pd.DataFrame(rows).sort_values(
        "expected_rank_improvement", ascending=False, kind="stable"
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result["recommendation"] = result["expected_rank_improvement"] > tie_tolerance
    return result.head(top_n)

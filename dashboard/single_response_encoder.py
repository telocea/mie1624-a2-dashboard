"""Encode one raw Stack Overflow survey response using fitted notebook metadata.

Run the final metadata-export cell in clean.ipynb first (see below), then:

    import joblib
    from single_response_encoder import encode_response
    encoded = encode_response(response_dict, joblib.load("survey_encoder.joblib"))

The returned one-row DataFrame is ordered exactly like the model's training X.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _norm(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    value = str(value).replace("’", "'").strip()
    return value or None


def _split(value: Any) -> list[str]:
    value = _norm(value)
    return [] if value is None else [v.strip() for v in value.split(";") if v.strip()]


def encode_response(response: Mapping[str, Any], metadata: Mapping[str, Any]) -> pd.DataFrame:
    """Transform one raw survey response to the fitted model's design matrix.

    `response` is a mapping from original survey column names to raw answers.
    Unknown categorical and multi-select values safely map to the fitted baseline
    (all-zero dummy representation); unknown ordinal values become missing and
    are imputed with the training median.
    """
    e = metadata
    row = {str(k): v for k, v in response.items()}
    out: dict[str, float] = {}
    normalize = lambda v: _norm(v)

    # Paired JobEase / JobDifficulty follow-ups collapse to their selected answer.
    ai_role = {
        "No, AI or AI agents were not a reason": "not a reason",
        "Yes, AI or AI agents contributed, but were not the main reason": "contributed",
        "Yes, AI or AI agents were the main reason": "main reason",
    }
    for base in ("JobEase", "JobDifficulty"):
        value = next((normalize(row.get(f"{base}AI_{i}")) for i in range(1, 11)
                      if normalize(row.get(f"{base}AI_{i}")) is not None), None)
        row[f"{base}_AI"] = ai_role.get(value) if value != "I'm not sure" else None
        for i in range(1, 11):
            row.pop(f"{base}AI_{i}", None)

    # AI tool selection and frequency follow-ups.
    # Backward compatible with metadata saved before ai_tool_short was added.
    tool_short = e.get("ai_tool_short", {
        "AI coding assistants or coding agents": "coding_assistants",
        "General-purpose AI chat tools": "chat",
        "AI agents or automated workflows": "agents",
        "Internal AI tools built by my company": "internal",
    })
    freq_order = e["freq_order"]
    selected_tools = set(_split(row.get("AISelect")))
    ai_section_answered = normalize(row.get("AISelect")) is not None
    for tool, short in tool_short.items():
        owner_col = e["ai_freq_owner"].get(short)
        freq = normalize(row.get(owner_col)) if owner_col else None
        val = (freq if tool in selected_tools else "Not used") if ai_section_answered else None
        # This is an ordered field later mapped through the notebook's ORDINAL
        # vocabulary, so preserve its category label here.
        row[f"AIFreq_{short}"] = val

    # Knowledge question: the fitted short-name-to-label/index mapping preserves
    # the original follow-up pairing used during training.
    knowledge_options = e["knowledge_options"]
    know_short = e["knowledge_short"]
    chosen = set(_split(row.get("KnowledgeQ")))
    knowledge_answered = normalize(row.get("KnowledgeQ")) is not None
    # The notebook creates selection flags for every observed KnowledgeQ option,
    # including options without an Acc/Time/Source follow-up column. Ratings
    # below are limited to `knowledge_options`, which has the follow-up pairing.
    for option, short in know_short.items():
        # Match cell 56: individual flags use the `sel__` prefix, while the
        # None/Not sure options are combined into one indicator below.
        if short not in e["knowledge_none_unsure"]:
            out[f"sel__{short}"] = float(option in chosen) if knowledge_answered else 0.0
    for short, option in knowledge_options.items():
        for fam, raw_family in (("Acc", "KnowledgeAcc"), ("Time", "KnowledgeTime"), ("Source", "KnowledgeSource")):
            # follow-up index mapping is explicitly exported from the fitted data.
            idx = e["knowledge_index"].get(short)
            raw_val = normalize(row.get(f"{raw_family}_{idx}")) if idx is not None else None
            levels = e["knowledge_levels"]
            if not knowledge_answered:
                label = None
            elif option not in chosen:
                label = "Not used"
            else:
                label = raw_val
            # Center ratings around Somewhat important; Not used and missing -> 0.
            code = levels.index(label) if label in levels else -1
            out[f"{fam}__{short}"] = float(code - levels.index("Somewhat important")) if code >= 1 else 0.0
    out["sel__none_or_unsure"] = float(any(x in chosen for x in e["knowledge_none_unsure"]))
    out["know_not_answered"] = float(not knowledge_answered)
    row.pop("KnowledgeQ", None)
    for c in list(row):
        if re.fullmatch(r"Knowledge(?:Acc|Time|Source)_\d+", c):
            row.pop(c, None)

    # Numeric and ordered fields. Clip/fence values exactly as in training.
    raw_numeric = e["numeric_columns"]
    ordinal_maps = e["ordinal_maps"]
    quant: dict[str, float] = {}
    for c in raw_numeric:
        value = row.get(c)
        try:
            value = float(value) if _norm(value) is not None else np.nan
        except (TypeError, ValueError):
            value = np.nan
        if c in ("YearsCode", "WorkExp") and np.isfinite(value):
            value = min(value, 50)
        if c in ("ToolCount", "AIToolCount") and value > 50:
            value = np.nan
        quant[c] = value
    for c, mapping in ordinal_maps.items():
        quant[c] = float(mapping.get(normalize(row.get(c)), np.nan))
    for c, value in list(quant.items()):
        if c in e["missing_indicator_columns"]:
            out[f"{c}__missing"] = float(pd.isna(value))
        value = e["medians"].get(c, 0.0) if pd.isna(value) else value
        if c in ("YearsCode", "WorkExp"):
            out[f"{c}_log"] = float(np.log1p(value))
        elif c != "ConvertedCompYearly":
            out[c] = float(value)

    # Nominal categories use fitted rare pools, missing handling and dummy vocab.
    nominal = e["nominal_columns"]
    for c in nominal:
        value = normalize(row.get(c)) or "Missing"
        if value in ("I prefer not to say", "Prefer not to say"):
            value = "Missing"
        if c == "Gender" and value in ("Non-binary", "Prefer to self-describe:"):
            value = "Other"
        if c in e["nominal_rare"] and value in e["nominal_rare"][c]:
            value = "Other"
        # unknown levels map to the reference category, yielding all zeroes
        known = e["nominal_levels"].get(c, [])
        if value not in known:
            value = e["nominal_reference"][c]
        for level in known:
            if level != e["nominal_reference"][c]:
                out[f"{c}__{level}"] = float(level == value)

    # Multi-select answers become fitted-vocabulary binary indicators.
    for c, levels in e["multi_levels"].items():
        values = set(_split(row.get(c)))
        for level in levels:
            out[f"{c}__{level}"] = float(level in values)

    # Notebook sanitizes dummy names after concatenation.
    out = {k.replace(" ", "_").replace(",", "").replace("/", "_"): v for k, v in out.items()}

    # Engineered predictors: use the training country pay table and medians.
    country = normalize(row.get("Country")) or "Missing"
    out["E_country_pay_idx"] = float(e["country_means"].get(country, e["country_global_mean"]))
    levels = [e["purchase_level"].get(x, -1) for x in _split(row.get("PurchasePower"))]
    authority = max(levels, default=np.nan)
    out["E_tool_authority"] = float(e["purchase_median"] if pd.isna(authority) or authority < 0 else authority)
    tech_counts = []
    for c in e["tech_sections"]:
        value = normalize(row.get(c))
        if value is not None:
            tech_counts.append(len(_split(value)))
    missing = not tech_counts
    tb = np.log1p(np.mean(tech_counts)) if tech_counts else e["tech_breadth_median"]
    out["E_tech_breadth__missing"] = float(missing)
    out["E_tech_breadth"] = float(tb)

    columns = e["training_columns"]
    result = pd.DataFrame([{c: out.get(c, 0.0) for c in columns}], columns=columns)
    # Training-time imputation covers modeled numeric/ordinal values. A 0 here
    # is for absent dummy/flag features, including fields not supplied by caller.
    result = result.fillna(0.0)
    return result.astype(float)

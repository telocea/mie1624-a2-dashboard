"""Interactive dashboard for the MIE1624 ordinal salary model.

Launch with: streamlit run dashboard_app.py
Place q4_final_model.joblib and survey_encoder.joblib beside this file.
"""
from __future__ import annotations

import json
import html
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import isotonic_regression
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d

from single_response_encoder import encode_response
from salary_skill_recommender import DEFAULT_CANDIDATE_PREFIXES

# q4_final_model.joblib was saved from a notebook where this estimator lived
# in __main__. Defining the same class name here lets joblib resolve that
# serialized class in this standalone Streamlit process.
class CumulativeOrdinalLogit(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1.0, l1_ratio=1, solver="saga", max_iter=1000, tol=1e-4,
                 class_weight=None, proba_floor=1e-4, decision="argmax",
                 categories=None, random_state=None):
        self.C = C
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.proba_floor = proba_floor
        self.decision = decision
        self.categories = categories
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Fit the cumulative binary classifiers, matching the notebook estimator."""
        X = check_array(X)
        y = column_or_1d(y, warn=True)
        categories = np.unique(y) if self.categories is None else np.asarray(self.categories)
        lookup = {value: i for i, value in enumerate(categories)}
        try:
            rank = np.array([lookup[value] for value in y])
        except KeyError as exc:
            raise ValueError(f"label {exc} not in categories") from None

        self.classes_ = categories
        self.n_features_in_ = X.shape[1]
        self.models_ = []
        self.constant_ = np.full(len(categories) - 1, np.nan)
        for threshold in range(len(categories) - 1):
            binary_target = (rank > threshold).astype(int)
            if binary_target.min() == binary_target.max():
                self.models_.append(None)
                self.constant_[threshold] = float(binary_target[0])
                continue
            classifier = LogisticRegression(
                C=self.C, l1_ratio=self.l1_ratio, solver=self.solver,
                max_iter=self.max_iter, tol=self.tol,
                class_weight=self.class_weight, random_state=self.random_state,
            )
            self.models_.append(
                classifier.fit(X, binary_target, sample_weight=sample_weight)
            )
        self.coef_ = np.vstack([
            estimator.coef_[0] if estimator is not None else np.zeros(X.shape[1])
            for estimator in self.models_
        ])
        self.intercept_ = np.array([
            estimator.intercept_[0] if estimator is not None else np.nan
            for estimator in self.models_
        ])
        return self

    def cumulative_proba(self, X):
        check_is_fitted(self, "models_")
        X = check_array(X)
        cols = [
            np.full(len(X), self.constant_[k]) if m is None else m.predict_proba(X)[:, 1]
            for k, m in enumerate(self.models_)
        ]
        return np.column_stack(cols)

    @staticmethod
    def _repair(G):
        G = np.clip(G, 0.0, 1.0)
        out = G.copy()
        for i in np.flatnonzero(np.any(np.diff(G, axis=1) > 1e-12, axis=1)):
            out[i] = isotonic_regression(y=G[i], y_min=0.0, y_max=1.0, increasing=False)
        return out

    @staticmethod
    def _to_class_proba(G):
        ext = np.hstack([np.ones((len(G), 1)), G, np.zeros((len(G), 1))])
        return ext[:, :-1] - ext[:, 1:]

    def predict_proba(self, X):
        G = self.cumulative_proba(X)
        K = G.shape[1] + 1
        P = np.clip(self._to_class_proba(self._repair(G)), 0.0, None)
        P /= P.sum(axis=1, keepdims=True)
        return (1 - K * self.proba_floor) * P + self.proba_floor

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "salary_model.joblib"
ENCODER_PATH = HERE / "survey_encoder.joblib"
RAW_COLUMNS_PATH = HERE / "consolidated_raw_columns.json"
QUESTION_SCHEMA_PATH = HERE / "dashboard_question_schema.json"
SALARY_LABELS = [
    "<$10k", "$10–20k", "$20–30k", "$30–40k", "$40–50k", "$50–75k",
    "$75–100k", "$100–125k", "$125–150k", "$150–200k", "$200–250k", "≥$250k",
]

st.set_page_config(page_title="Salary distribution explorer", page_icon="📊", layout="wide")
st.title("Salary distribution explorer")
st.caption(
    "Survey responses are encoded with the training-fitted encoder, then passed "
    "through the final ordinal model. Estimates are associations, not causal salary forecasts."
)

if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
    st.error("The dashboard needs salary_model.joblib and survey_encoder.joblib beside dashboard_app.py.")
    st.info("See DASHBOARD_README.md for the one-time encoder export step from clean.ipynb.")
    st.stop()

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

model, encoder = load_artifacts()
model_columns = list(getattr(model, "feature_names_in_", []))
encoder_columns = list(encoder.get("training_columns", []))
if not model_columns or model_columns != encoder_columns:
    st.error(
        "The saved model and encoder do not have the same training columns. "
        "Export salary_model.joblib and survey_encoder.joblib from the same notebook run."
    )
    st.stop()
raw_columns = json.loads(RAW_COLUMNS_PATH.read_text())["raw_csv_columns"]
question_schema = json.loads(QUESTION_SCHEMA_PATH.read_text())["questions"]
# This is the source-column set used by the selected model inputs. Pay/compensation
# fields and identifiers are intentionally omitted.
excluded = {"ResponseId", "Currency", "CompTotal", "ConvertedCompYearly", "ConvertedCompYearlyBucket"}
raw_columns = [c for c in raw_columns if c not in excluded]

def question_text(column: str) -> str:
    record = question_schema.get(column, {})
    question = html.unescape(record.get("question", column))
    sub = html.unescape(record.get("sub", "NA"))
    if sub and sub != "NA":
        return f"{question}\n\n**{sub}**"
    return question

def raw_choices(column: str) -> list[str]:
    opts: list[str] = []
    opts.extend(encoder.get("nominal_levels", {}).get(column, []))
    opts.extend(encoder.get("nominal_rare", {}).get(column, []))
    if column in encoder.get("ordinal_maps", {}):
        opts.extend(encoder["ordinal_maps"][column].keys())
    if column in encoder.get("multi_levels", {}):
        opts.extend(encoder["multi_levels"][column])
    if column.startswith("AIFreq_"):
        opts.extend(encoder.get("freq_order", []))
    if column == "KnowledgeQ":
        opts.extend(encoder.get("knowledge_short", {}).keys())
    return list(dict.fromkeys(str(x) for x in opts if x and str(x) not in ("Missing", "nan")))

def make_widget(column: str):
    st.markdown(question_text(column))
    if column in encoder.get("numeric_columns", []):
        if st.checkbox("I don't know / prefer not to answer", key=f"{column}_missing"):
            return None
        maximum = 50 if column in ("YearsCode", "WorkExp", "ToolCount", "AIToolCount") else 100
        return st.number_input("Your answer", min_value=0.0, max_value=float(maximum), value=0.0, step=1.0, key=column)
    if column in encoder.get("ordinal_maps", {}):
        choices = ["No response"] + list(encoder["ordinal_maps"][column].keys())
        value = st.selectbox("Your answer", choices, key=column)
        return None if value == "No response" else value
    if column in encoder.get("multi_levels", {}):
        choices = raw_choices(column)
        selected = st.multiselect("Select all that apply", choices, key=column)
        return ";".join(selected) if selected else None
    if column == "KnowledgeQ":
        choices = raw_choices(column)
        selected = st.multiselect("Select up to 5", choices, key=column)
        return ";".join(selected) if selected else None
    if column in encoder.get("nominal_columns", []):
        choices = ["No response"] + raw_choices(column)
        selected = st.selectbox("Your answer", choices, key=column)
        return None if selected == "No response" else selected
    if column.startswith("AIFreq_"):
        choices = ["No response"] + encoder.get("freq_order", [])
        selected = st.selectbox("Your answer", choices, key=column)
        return None if selected == "No response" else selected
    # Several notebook follow-up columns are only relevant when a parent
    # multi-select option is chosen. A blank answer is a valid raw response.
    return None

sections = {
    "About you and work": ["Age", "Country", "Education", "Employment", "WorkExp", "YearsCode",
                           "Role", "Industry", "OrgSize", "WorkSetting", "ICorPM", "CodeMain", "Gender"],
    "Skills and tools": [c for c in raw_columns if any(k in c for k in (
        "HaveWorkedWith", "WantToWorkWith", "LanguagesUsedAI", "BuildTool", "PurchasePower",
        "ToolCount", "TechBudget", "TechDropDown"))],
    "AI and work practices": [c for c in raw_columns if any(k in c for k in (
        "AI", "JobEase", "JobDifficulty", "JobSat", "Knowledge", "Context", "Answers", "Trust",
        "Disruption", "RoleFunctions", "Community", "Learn", "SO"))],
}
# A column can match more than one keyword group (for example,
# AICodeAgentHaveWorkedWith matches both "AI" and "HaveWorkedWith"). Assign
# each raw question to the first matching tab only, so Streamlit widget keys
# stay unique.
assigned = set()
for title, columns in sections.items():
    unique_columns = []
    for column in columns:
        if column in raw_columns and column not in assigned:
            unique_columns.append(column)
            assigned.add(column)
    sections[title] = unique_columns
sections["Other selected survey questions"] = [c for c in raw_columns if c not in assigned]

with st.form("survey_form"):
    st.markdown("### Answer the selected survey questions")
    st.caption("Complete the questions across the tabs, then submit once. The salary distribution is calculated only after you press the button below. Leave an item unanswered when appropriate.")
    answers = {}
    tabs = st.tabs(list(sections))
    for (title, columns), tab in zip(sections.items(), tabs):
        with tab:
            for column in columns:
                value = make_widget(column)
                if value is not None:
                    answers[column] = value
                if column in ("JobEase", "JobDifficulty") and value is not None:
                    # Ask the schema's conditional AI follow-up for the factor
                    # the respondent selected, then pass it through its actual
                    # corresponding raw follow-up column.
                    options = ["No response"] + [
                        "No, AI or AI agents were not a reason",
                        "Yes, AI or AI agents contributed, but were not the main reason",
                        "Yes, AI or AI agents were the main reason",
                        "I'm not sure",
                    ]
                    matches = [
                        (f"{column}AI_{i}", question_schema.get(f"{column}AI_{i}", {}))
                        for i in range(1, 11)
                    ]
                    match = next(
                        ((field, meta) for field, meta in matches
                         if html.unescape(meta.get("sub", "")).strip() == str(value).strip()),
                        None,
                    )
                    if match:
                        followup_field, _ = match
                        st.markdown(question_text(followup_field))
                        followup = st.selectbox(
                            "Your answer", options, key=f"{followup_field}_answer"
                        )
                        if followup != "No response":
                            answers[followup_field] = followup
    submitted = st.form_submit_button("Predict my salary bucket")

if submitted:
    try:
        encoded = encode_response(answers, encoder).reindex(columns=model_columns, fill_value=0.0)
        probabilities = np.asarray(model.predict_proba(encoded)[0], dtype=float)
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = np.arange(len(probabilities))
        if len(probabilities) != len(SALARY_LABELS):
            st.error(f"Expected 12 salary classes, got {len(probabilities)}.")
            st.stop()
        result = pd.DataFrame({"Salary bucket": SALARY_LABELS, "Probability": probabilities})
        left, right = st.columns([1.15, 1])
        with left:
            st.subheader("Predicted salary-bucket distribution")
            fig, ax = plt.subplots(figsize=(10, 4.3))
            palette = ["#E0A82E" if i == int(np.argmax(probabilities)) else "#4C78A8"
                       for i in range(len(probabilities))]
            ax.bar(result["Salary bucket"], result["Probability"], color=palette)
            ax.set_ylabel("Predicted probability")
            ax.set_ylim(0, max(0.05, probabilities.max() * 1.2))
            ax.tick_params(axis="x", rotation=35)
            ax.grid(axis="y", alpha=.2)
            st.pyplot(fig, clear_figure=True)
            mode = int(np.argmax(probabilities))
            median = int(np.searchsorted(np.cumsum(probabilities), .5))
            st.write(f"**Most likely:** {SALARY_LABELS[mode]} ({probabilities[mode]:.1%})  ·  "
                     f"**Median bucket:** {SALARY_LABELS[median]}")
        with right:
            st.subheader("Ranked learning recommendations")
            model_cols = model_columns
            respondent = encoded.loc[:, model_cols]
            base_rank = float(probabilities @ np.arange(len(probabilities)))
            recs = []
            for feature in model_cols:
                if not feature.startswith(DEFAULT_CANDIDATE_PREFIXES):
                    continue
                if pd.to_numeric(respondent.iloc[0][feature], errors="coerce") != 0:
                    continue
                scenario = respondent.copy()
                scenario.loc[scenario.index[0], feature] = 1.0
                p = model.predict_proba(scenario)[0]
                recs.append((float(p @ np.arange(len(p))) - base_rank, feature.split("__", 1)[1]))
            recs.sort(key=lambda x: x[0], reverse=True)
            positive = [(score, skill) for score, skill in recs if score > 1e-12]
            if positive:
                st.dataframe(pd.DataFrame([
                    {"Choice": i, "Skill / technology / tool": skill, "Expected bucket-rank improvement": f"+{score:.3f}"}
                    for i, (score, skill) in enumerate(positive, 1)
                ]), hide_index=True, use_container_width=True)
            else:
                st.info("The fitted model found no positive one-feature improvement among its supported candidate tools.")
        st.caption(
            "Improvement is the change in expected ordered bucket index after changing one currently absent "
            "skill/tool indicator from 0 to 1, with other answers held fixed. It is a model association and "
            "does not show that learning the skill causes higher pay."
        )
    except Exception as exc:
        st.exception(exc)

with st.expander("How to read this result"):
    st.write(
        "Probabilities describe the model's uncertainty across the 12 ordered salary categories. "
        "Survey associations can reflect country, experience, role, education, selection effects, and other "
        "factors. Recommendations are hypothetical, one-at-a-time model scenarios; they do not account for "
        "learning time, proficiency, job demand, or interactions among skills."
    )

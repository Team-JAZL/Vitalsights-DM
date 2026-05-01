import hashlib
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DEFAULT_DATA_PATH = DATA_DIR / "heart.csv"
NOTEBOOK_PATH = BASE_DIR / "vitalsights_combined.ipynb"

TARGET_COLUMN = "HeartDisease"
PREDICTION_COLUMNS = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]
REQUIRED_COLUMNS = PREDICTION_COLUMNS + [TARGET_COLUMN]
CATEGORICAL_COLUMNS = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]
NUMERICAL_FEATURES = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
RISK_BINS = [0.0, 0.05, 0.075, 0.20, 1.0000001]
RISK_LABELS = ["Low Risk", "Borderline Risk", "Intermediate Risk", "High Risk"]
ACTION_MAP = {
    "Low Risk": "Routine monitoring",
    "Borderline Risk": "Lifestyle counseling and follow-up",
    "Intermediate Risk": "Further diagnostic evaluation",
    "High Risk": "Immediate clinical review or aggressive intervention",
}
MARKDOWN_COMMENTARY_MARKERS = (
    "***OBSERVATION:***",
    "***EXPLANATION:***",
    "***ANALYSIS:***",
    "***TRADE-OFF:",
)
PAGES = [
    "Home / Project Overview",
    "Data Upload and Cleaning",
    "Exploratory Data Analysis",
    "Predictive Modeling",
    "Model Interpretability",
    "Clinical Risk Stratification",
    "Apriori Association Rules",
    "K-Means Clustering",
    "New Patient Prediction",
]

sns.set_theme(style="whitegrid")


st.set_page_config(
    page_title="VitalSights Data Mining Dashboard",
    page_icon="VS",
    layout="wide",
)


@st.cache_data
def load_default_dataset() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_DATA_PATH)


@st.cache_data
def load_notebook_markdown() -> list[str]:
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    ]


def show_notebook_markdown_cell(index: int):
    markdown_cells = load_notebook_markdown()
    if index < len(markdown_cells):
        text = markdown_cells[index].strip()
        if (
            text
            and set(text.replace("\n", "")) != {"-"}
            and not any(marker in text for marker in MARKDOWN_COMMENTARY_MARKERS)
        ):
            st.markdown(text)


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def set_active_training_dataset(raw_df: pd.DataFrame, source: str):
    cleaned_df, cleaning_summary = clean_heart_data(raw_df)
    st.session_state["active_raw_df"] = raw_df.copy()
    st.session_state["active_cleaned_df"] = cleaned_df.copy()
    st.session_state["active_cleaning_summary"] = cleaning_summary.copy()
    st.session_state["active_source"] = source
    st.session_state["active_fingerprint"] = dataframe_fingerprint(raw_df)


def initialize_active_dataset():
    if "active_cleaned_df" not in st.session_state:
        set_active_training_dataset(load_default_dataset(), "data/heart.csv")


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def validate_columns(df: pd.DataFrame, training_mode: bool = True) -> list[str]:
    expected = REQUIRED_COLUMNS if training_mode else PREDICTION_COLUMNS
    return [col for col in expected if col not in df.columns]


@st.cache_data
def clean_heart_data(df: pd.DataFrame, medians: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = df.copy()
    before_after = []
    for column in ["RestingBP", "Cholesterol"]:
        before_min = pd.to_numeric(cleaned[column], errors="coerce").min()
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").replace(0, np.nan)
        fill_value = medians[column] if medians and column in medians else cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(fill_value)
        after_min = cleaned[column].min()
        before_after.append(
            {
                "Column": column,
                "Minimum Before Cleaning": before_min,
                "Median Used": fill_value,
                "Minimum After Cleaning": after_min,
            }
        )
    return cleaned, pd.DataFrame(before_after)


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    cat_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return pd.get_dummies(df, columns=cat_columns, drop_first=True, dtype=int)


def align_prediction_columns(encoded_df: pd.DataFrame, model_columns: list[str]) -> pd.DataFrame:
    aligned = encoded_df.copy()
    for column in model_columns:
        if column not in aligned.columns:
            aligned[column] = 0
    return aligned[model_columns]


def apply_risk_tiers(probabilities: pd.Series | np.ndarray) -> pd.Series:
    return pd.cut(
        pd.Series(probabilities),
        bins=RISK_BINS,
        labels=RISK_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)


@st.cache_resource
def train_predictive_models(cleaned_df: pd.DataFrame) -> dict:
    encoded = encode_features(cleaned_df)
    y = encoded[TARGET_COLUMN].copy()
    X = encoded.drop(columns=[TARGET_COLUMN]).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        class_weight="balanced",
        random_state=42,
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Decision Tree",
                "Accuracy": accuracy_score(y_test, y_pred_dt),
                "Precision": precision_score(y_test, y_pred_dt),
                "Recall": recall_score(y_test, y_pred_dt),
            },
            {
                "Model": "Random Forest",
                "Accuracy": accuracy_score(y_test, y_pred_rf),
                "Precision": precision_score(y_test, y_pred_rf),
                "Recall": recall_score(y_test, y_pred_rf),
            },
        ]
    )

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(rf_model, MODEL_DIR / "random_forest_model.pkl")
    joblib.dump(dt_model, MODEL_DIR / "decision_tree_model.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "model_columns.pkl")

    return {
        "encoded": encoded,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "dt_model": dt_model,
        "rf_model": rf_model,
        "y_pred_dt": y_pred_dt,
        "y_pred_rf": y_pred_rf,
        "y_proba_rf": y_proba_rf,
        "metrics_df": metrics_df,
        "model_columns": list(X.columns),
    }


def prepare_apriori_frame(df: pd.DataFrame) -> pd.DataFrame:
    mined = df.copy()
    mined["Age_Group"] = pd.cut(
        mined["Age"], bins=[0, 45, 60, 100], labels=["Young", "Middle-Aged", "Senior"]
    )
    mined["BP_Level"] = pd.cut(
        mined["RestingBP"],
        bins=[0, 120, 140, 300],
        labels=["Normal_BP", "Pre-Hypertension", "Hypertension"],
    )
    mined["Chol_Level"] = pd.cut(
        mined["Cholesterol"],
        bins=[0, 200, 240, 700],
        labels=["Normal_Chol", "Borderline_High", "High_Chol"],
    )
    mined["MaxHR_Level"] = pd.cut(
        mined["MaxHR"], bins=[0, 120, 155, 220], labels=["Low_MaxHR", "Avg_MaxHR", "High_MaxHR"]
    )
    mined["ST_Depression"] = pd.cut(
        mined["Oldpeak"],
        bins=[-5, 0.5, 2.0, 10],
        labels=["Low_ST_Dep", "Moderate_ST_Dep", "Severe_ST_Dep"],
    )
    if TARGET_COLUMN in mined.columns:
        mined["HeartDisease_Lab"] = mined[TARGET_COLUMN].map(
            {1: "Has_HeartDisease", 0: "No_HeartDisease"}
        )
    return mined


def mining_columns(include_target: bool = True) -> list[str]:
    columns = [
        "Sex",
        "ChestPainType",
        "Age_Group",
        "BP_Level",
        "Chol_Level",
        "MaxHR_Level",
        "ST_Depression",
        "FastingBS",
        "ExerciseAngina",
        "ST_Slope",
    ]
    if include_target:
        columns.append("HeartDisease_Lab")
    return columns


def prune_redundant_rules(risk_rules: pd.DataFrame) -> pd.DataFrame:
    if risk_rules.empty:
        return risk_rules

    def is_redundant(rule, all_rules):
        for other_index, other in all_rules.iterrows():
            if rule.name == other_index:
                continue
            if set(other["antecedents"]).issubset(set(rule["antecedents"])):
                if rule["confidence"] <= other["confidence"]:
                    return True
        return False

    pruned = risk_rules.copy()
    pruned["is_redundant"] = pruned.apply(lambda row: is_redundant(row, pruned), axis=1)
    return pruned[~pruned["is_redundant"]].drop(columns=["is_redundant"]).reset_index(drop=True)


@st.cache_data
def run_apriori(cleaned_df: pd.DataFrame) -> dict:
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError as exc:
        return {"error": f"mlxtend is required for Apriori mining: {exc}"}

    mined = prepare_apriori_frame(cleaned_df)
    df_encoded = pd.get_dummies(mined[mining_columns(include_target=True)])
    frequent_itemsets = apriori(df_encoded.astype(bool), min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    risk_rules = rules[
        rules["consequents"].apply(lambda x: "HeartDisease_Lab_Has_HeartDisease" in set(x))
    ]
    risk_rules = risk_rules.sort_values(by="lift", ascending=False).reset_index(drop=True)
    pruned_rules = prune_redundant_rules(risk_rules)
    rulebook = make_rulebook(pruned_rules)
    return {
        "mined": mined,
        "encoded": df_encoded,
        "frequent_itemsets": frequent_itemsets,
        "rules": rules,
        "risk_rules": risk_rules,
        "pruned_rules": pruned_rules,
        "rulebook": rulebook,
    }


def format_itemset(itemset) -> str:
    return ", ".join(sorted([str(item) for item in itemset]))


def make_rulebook(pruned_rules: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    rows = []
    rules = pruned_rules if limit is None else pruned_rules.head(limit)
    for idx, row in rules.reset_index(drop=True).iterrows():
        antecedents = format_itemset(row["antecedents"])
        rows.append(
            {
                "Rule": idx + 1,
                "Antecedents": antecedents,
                "Confidence": row["confidence"],
                "Lift": row["lift"],
                "Plain English Rule": (
                    f"IF a patient presents with [{antecedents}], THEN the probability of "
                    f"Heart Disease is {row['confidence'] * 100:.1f}%. This profile is "
                    f"{row['lift']:.2f}x more likely than the average patient."
                ),
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource
def train_kmeans(cleaned_df: pd.DataFrame) -> dict:
    X = cleaned_df[NUMERICAL_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=NUMERICAL_FEATURES)

    wcss = []
    for k in range(1, 11):
        candidate = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        candidate.fit(X_scaled)
        wcss.append(candidate.inertia_)

    kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=10)
    clustered_df = cleaned_df.copy()
    clustered_df["Cluster"] = kmeans.fit_predict(X_scaled)
    cluster_profile = clustered_df.groupby("Cluster")[NUMERICAL_FEATURES + [TARGET_COLUMN]].mean()
    cluster_profile["Risk_Label"] = cluster_profile[TARGET_COLUMN].apply(cluster_risk_label)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(kmeans, MODEL_DIR / "kmeans_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    return {
        "scaler": scaler,
        "X_scaled": X_scaled,
        "X_scaled_df": X_scaled_df,
        "wcss": wcss,
        "kmeans": kmeans,
        "clustered_df": clustered_df,
        "cluster_profile": cluster_profile,
    }


def cluster_risk_label(prevalence: float) -> str:
    if prevalence > 0.75:
        return "CRITICAL RISK"
    if prevalence > 0.50:
        return "HIGH RISK"
    return "LOW/MODERATE RISK"


def predict_patients(
    input_df: pd.DataFrame,
    cleaned_training_df: pd.DataFrame,
    model_bundle: dict,
    cluster_bundle: dict,
    apriori_bundle: dict | None = None,
) -> pd.DataFrame:
    medians = cleaned_training_df[["RestingBP", "Cholesterol"]].median().to_dict()
    cleaned_input, _ = clean_heart_data(input_df[PREDICTION_COLUMNS], medians=medians)
    encoded = encode_features(cleaned_input)
    X_pred = align_prediction_columns(encoded, model_bundle["model_columns"])
    rf_model = model_bundle["rf_model"]

    probabilities = rf_model.predict_proba(X_pred)[:, 1]
    predictions = rf_model.predict(X_pred)
    result = input_df.copy()
    result["Predicted_Status"] = predictions
    result["Predicted_Label"] = np.where(predictions == 1, "Heart Disease", "Normal")
    result["Heart_Disease_Probability"] = probabilities
    result["Heart_Disease_Probability_Percent"] = (probabilities * 100).round(1)
    result["ACC_AHA_Risk_Tier"] = apply_risk_tiers(probabilities)
    result["Clinical_Action"] = result["ACC_AHA_Risk_Tier"].map(ACTION_MAP)

    scaled = cluster_bundle["scaler"].transform(cleaned_input[NUMERICAL_FEATURES])
    clusters = cluster_bundle["kmeans"].predict(scaled)
    result["Clinical_Cluster"] = clusters
    profile = cluster_bundle["cluster_profile"]
    result["Cluster_Risk_Interpretation"] = [
        profile.loc[cluster, "Risk_Label"] if cluster in profile.index else "Unknown"
        for cluster in clusters
    ]

    if apriori_bundle and not apriori_bundle.get("error") and not apriori_bundle["pruned_rules"].empty:
        matched_rules = []
        for _, patient in cleaned_input.iterrows():
            matched_rules.append(match_patient_rules(patient, apriori_bundle["pruned_rules"]))
        result["Matched_Apriori_Rules"] = matched_rules
    return result


def patient_mining_items(patient: pd.Series) -> set[str]:
    patient_df = pd.DataFrame([patient])
    prepared = prepare_apriori_frame(patient_df)
    encoded = pd.get_dummies(prepared[mining_columns(include_target=False)])
    return set(encoded.columns[encoded.iloc[0].astype(bool)].tolist())


def match_patient_rules(patient: pd.Series, pruned_rules: pd.DataFrame, limit: int = 3) -> str:
    items = patient_mining_items(patient)
    matches = []
    for _, row in pruned_rules.iterrows():
        antecedents = set(row["antecedents"])
        if antecedents.issubset(items):
            matches.append(
                f"{format_itemset(antecedents)} | confidence {row['confidence'] * 100:.1f}%, "
                f"lift {row['lift']:.2f}"
            )
    return " || ".join(matches[:limit]) if matches else "No top pruned rules matched"


def show_interpretation(text: str):
    st.info(text)


def render_metric_cards(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Heart Disease Rate", f"{df[TARGET_COLUMN].mean() * 100:.1f}%" if TARGET_COLUMN in df else "N/A")
    c4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")


def fig_null_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Null Value Heatmap")
    return fig


def fig_univariate(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 3, figsize=(18, 18))
    axes = axes.flatten()
    features = REQUIRED_COLUMNS
    colors = [
        "skyblue",
        "mediumseagreen",
        "lightcoral",
        "salmon",
        "lightgreen",
        "plum",
        "khaki",
        "orchid",
        "lightsalmon",
        "gold",
        "cornflowerblue",
        "lightsteelblue",
    ]
    countplot_palettes = {
        "Sex": "Set2",
        "ChestPainType": "Set3",
        "FastingBS": "Set1",
        "RestingECG": "Set2",
        "ExerciseAngina": "Set3",
        "ST_Slope": "Set1",
        "HeartDisease": "Set2",
    }
    for ax, feature, color in zip(axes, features, colors):
        if pd.api.types.is_numeric_dtype(df[feature]) and feature not in ["FastingBS", TARGET_COLUMN]:
            sns.histplot(df[feature], bins=20, kde=True, color=color, ax=ax)
            ax.axvline(df[feature].mean(), color="red", linestyle="--", label=f"Mean {df[feature].mean():.1f}")
            ax.axvline(df[feature].median(), color="green", linestyle="--", label=f"Median {df[feature].median():.1f}")
            ax.legend(fontsize=8)
        else:
            sns.countplot(
                data=df,
                x=feature,
                hue=feature,
                palette=countplot_palettes.get(feature, "Set2"),
                legend=False,
                ax=ax,
            )
            ax.tick_params(axis="x", rotation=25)
        ax.set_title(f"{feature} Distribution")
    fig.tight_layout()
    return fig


def fig_kde_numeric(df: pd.DataFrame):
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(
        ["FastingBS", TARGET_COLUMN], errors="ignore"
    )
    fig, axes = plt.subplots(len(numerical_cols), 1, figsize=(12, len(numerical_cols) * 3))
    if len(numerical_cols) == 1:
        axes = [axes]
    for ax, feature in zip(axes, numerical_cols):
        sns.histplot(df[feature], kde=True, ax=ax)
        ax.set_title(f"{feature} | Skewness: {df[feature].skew():.2f}")
    fig.tight_layout()
    return fig


def fig_box_numeric(df: pd.DataFrame):
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(
        ["FastingBS", TARGET_COLUMN], errors="ignore"
    )
    fig, axes = plt.subplots(len(numerical_cols), 1, figsize=(12, len(numerical_cols) * 2.6))
    if len(numerical_cols) == 1:
        axes = [axes]
    for ax, feature in zip(axes, numerical_cols):
        sns.boxplot(x=df[feature], color="lightblue", ax=ax)
        ax.set_title(f"{feature} | Skewness: {df[feature].skew():.2f}")
    fig.tight_layout()
    return fig


def fig_cleaned_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, feature, color in [
        (axes[0], "RestingBP", "salmon"),
        (axes[1], "Cholesterol", "lightgreen"),
    ]:
        sns.histplot(df[feature], bins=20, kde=True, color=color, ax=ax)
        ax.axvline(df[feature].mean(), color="red", linestyle="--", label=f"Mean {df[feature].mean():.1f}")
        ax.axvline(df[feature].median(), color="green", linestyle="--", label=f"Median {df[feature].median():.1f}")
        ax.set_title(f"Cleaned {feature} Distribution")
        ax.legend()
    fig.tight_layout()
    return fig


def fig_categorical_by_target(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if "FastingBS" not in cat_cols:
        cat_cols.append("FastingBS")
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for ax, feature in zip(axes, cat_cols):
        sns.countplot(data=df, x=feature, hue=TARGET_COLUMN, palette="Set2", ax=ax)
        ax.set_title(f"{feature} vs Heart Disease")
        ax.tick_params(axis="x", rotation=25)
    for ax in axes[len(cat_cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def fig_numeric_by_target(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(
        ["FastingBS", TARGET_COLUMN], errors="ignore"
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for ax, feature in zip(axes, num_cols):
        sns.boxplot(data=df, x=TARGET_COLUMN, y=feature, hue=TARGET_COLUMN, palette="Set2", legend=False, ax=ax)
        ax.set_title(f"{feature} by Heart Disease")
    for ax in axes[len(num_cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def fig_correlation(df: pd.DataFrame):
    encoded = pd.get_dummies(df)
    corr = encoded.corr()
    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Full Dataset Correlation Matrix")
    fig.tight_layout()
    return fig


def fig_pairplot(df: pd.DataFrame):
    pairplot_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(["FastingBS"], errors="ignore")
    grid = sns.pairplot(data=df[pairplot_cols], hue=TARGET_COLUMN, palette="Set1", corner=True, diag_kind="kde")
    grid.fig.suptitle("Pairplot of Numerical Features by Heart Disease", y=1.02)
    return grid.fig


def fig_confusion_matrices(bundle: dict):
    cm_dt = confusion_matrix(bundle["y_test"], bundle["y_pred_dt"])
    cm_rf = confusion_matrix(bundle["y_test"], bundle["y_pred_rf"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(
        cm_dt,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal", "Heart Disease"],
        yticklabels=["Normal", "Heart Disease"],
        ax=axes[0],
    )
    axes[0].set_title("Decision Tree Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    sns.heatmap(
        cm_rf,
        annot=True,
        fmt="g",
        cmap="Greens",
        cbar=False,
        xticklabels=["Normal", "Heart Disease"],
        yticklabels=["Normal", "Heart Disease"],
        ax=axes[1],
    )
    axes[1].set_title("Random Forest Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    fig.tight_layout()
    return fig


def classification_report_table(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Heart Disease"],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    total_support = len(y_true)
    for label in ["Normal", "Heart Disease", "accuracy", "macro avg", "weighted avg"]:
        if label == "accuracy":
            rows.append(
                {
                    "Class / Average": label,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1-score": round(report[label], 4),
                    "support": total_support,
                }
            )
        else:
            rows.append(
                {
                    "Class / Average": label,
                    "precision": round(report[label]["precision"], 4),
                    "recall": round(report[label]["recall"], 4),
                    "f1-score": round(report[label]["f1-score"], 4),
                    "support": int(report[label]["support"]),
                }
            )
    return pd.DataFrame(rows)


def fig_decision_tree(model, feature_names, max_depth=None):
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Normal", "Heart Disease"],
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=max_depth,
        ax=ax,
    )
    return fig


def fig_feature_importance(bundle: dict):
    X = bundle["X"]
    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "Decision Tree Importance": bundle["dt_model"].feature_importances_,
            "Random Forest Importance": bundle["rf_model"].feature_importances_,
        }
    )
    dt_sorted = importance_df.sort_values(by="Decision Tree Importance", ascending=False)
    rf_sorted = importance_df.sort_values(by="Random Forest Importance", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.barplot(x="Decision Tree Importance", y="feature", data=dt_sorted, ax=axes[0], color="#4c72b0")
    sns.barplot(x="Random Forest Importance", y="feature", data=rf_sorted, ax=axes[1], color="#2ca02c")
    axes[0].set_title("Decision Tree Feature Importances")
    axes[1].set_title("Random Forest Feature Importances")
    axes[1].set_ylabel("")
    fig.tight_layout()
    return fig


def make_patient_risk_df(bundle: dict) -> pd.DataFrame:
    risk_df = bundle["X_test"].reset_index(drop=True).copy()
    risk_df["True_Status"] = bundle["y_test"].reset_index(drop=True)
    risk_df["Predicted_Status"] = bundle["y_pred_rf"]
    risk_df["Heart_Disease_Probability"] = bundle["y_proba_rf"]
    risk_df["ACC_AHA_Risk_Tier"] = apply_risk_tiers(bundle["y_proba_rf"])
    return risk_df


def fig_risk_tiers(patient_risk_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    sns.countplot(
        data=patient_risk_df,
        x="ACC_AHA_Risk_Tier",
        hue="ACC_AHA_Risk_Tier",
        order=RISK_LABELS,
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_title("Hospital Resource Allocation based on ACC/AHA Risk Tiers")
    ax.set_xlabel("ACC/AHA Risk Tier")
    ax.set_ylabel("Number of Patients")
    for patch in ax.patches:
        count = int(patch.get_height())
        ax.annotate(
            f"{count}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    return fig


def fig_apriori_bar(pruned_rules: pd.DataFrame):
    top = pruned_rules.head(10).copy()
    top["rule_label"] = top["antecedents"].apply(format_itemset)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="lift", y="rule_label", data=top, palette="Reds_d", hue="rule_label", legend=False, ax=ax)
    ax.set_title("Top 10 Risk Archetypes Ranked by Lift")
    ax.set_xlabel("Lift")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def fig_apriori_network(pruned_rules: pd.DataFrame):
    graph = nx.DiGraph()
    for _, row in pruned_rules.head(15).iterrows():
        for antecedent in row["antecedents"]:
            for consequent in row["consequents"]:
                graph.add_edge(str(antecedent), str(consequent), weight=row["lift"])
    fig, ax = plt.subplots(figsize=(16, 10))
    if graph.number_of_nodes() > 0:
        pos = nx.spring_layout(graph, k=0.7, seed=42)
        nx.draw_networkx_nodes(graph, pos, node_size=5500, node_color="lightblue", edgecolors="black", alpha=0.9, ax=ax)
        nx.draw_networkx_edges(graph, pos, width=1.5, edge_color="gray", arrowsize=20, alpha=0.55, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=7, font_weight="bold", ax=ax)
    ax.set_title("Map of Clinical Risk Pathways and Hub Red Flags")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_rule_strength_heatmap(pruned_rules: pd.DataFrame):
    heatmap_data = pruned_rules.copy()
    heatmap_data["Confidence_Range"] = pd.cut(heatmap_data["confidence"], bins=5).astype(str)
    heatmap_data["Support_Range"] = pd.cut(heatmap_data["support"], bins=5).astype(str)
    pivot_table = heatmap_data.pivot_table(
        index="Confidence_Range",
        columns="Support_Range",
        values="lift",
        aggfunc="max",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot_table, annot=True, cmap="YlOrRd", fmt=".2f", ax=ax)
    ax.set_title("Statistical Strength Heatmap (Color = Lift)")
    fig.tight_layout()
    return fig


def fig_elbow(wcss: list[float]):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, 11), wcss, marker="o", linestyle="--", color="#1f77b4")
    ax.set_title("The Elbow Method: Finding Optimal Patient Clusters")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.grid(True, alpha=0.3)
    return fig


def fig_cluster_scatter(clustered_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=clustered_df,
        x="Age",
        y="MaxHR",
        hue="Cluster",
        palette="viridis",
        style=TARGET_COLUMN,
        s=90,
        alpha=0.8,
        ax=ax,
    )
    ax.set_title("Data Mining Result: Patient Segmentation Map")
    ax.set_xlabel("Patient Age")
    ax.set_ylabel("Max Heart Rate Achieved")
    ax.legend(title="Cluster / Heart Disease", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


def render_home(cleaned_df: pd.DataFrame, source: str):
    show_notebook_markdown_cell(0)
    st.title("VitalSights Dashboard")
    st.write(
        "VitalSights is a Streamlit data mining dashboard for heart disease risk prediction, "
        "pattern discovery, and clinical decision-support prototyping."
    )
    if st.button("Go to New Patient Prediction", type="primary"):
        st.session_state["page"] = "New Patient Prediction"
        st.rerun()
    st.caption(f"Active dashboard dataset: {source}")
    render_metric_cards(cleaned_df)
    st.subheader("Workflow")
    st.write(
        "The application follows the notebook pipeline: Data Loading, Data Cleaning, EDA, "
        "Predictive Modeling, Model Interpretability, Clinical Risk Stratification, Apriori "
        "Association Rule Mining, K-Means Clustering, and New Patient Prediction."
    )
    st.subheader("Dataset")
    st.write(
        "The active training dataset contains patients, clinical features, and a binary HeartDisease "
        "target where 0 means Normal / No Heart Disease and 1 means Heart Disease."
    )
    st.dataframe(cleaned_df.head(10), use_container_width=True)


def render_upload_cleaning():
    st.title("Data Upload and Cleaning")
    uploaded = st.file_uploader("Upload CSV or Excel data", type=["csv", "xlsx", "xls"])
    mode = st.radio("Validation mode", ["Training data", "Prediction data"], horizontal=True)
    st.caption(f"Active dashboard dataset: {st.session_state.get('active_source', 'data/heart.csv')}")
    if st.button("Reset dashboard to default dataset"):
        set_active_training_dataset(load_default_dataset(), "data/heart.csv")
        st.rerun()

    if uploaded:
        raw_df = read_uploaded_table(uploaded)
        source = uploaded.name
    else:
        raw_df = st.session_state.get("active_raw_df", load_default_dataset())
        source = st.session_state.get("active_source", "data/heart.csv")

    st.caption(f"Current source: {source}")
    missing = validate_columns(raw_df, training_mode=(mode == "Training data"))
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
    else:
        st.success("Column validation passed.")

    st.subheader("Raw Dataset Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)
    render_metric_cards(raw_df if TARGET_COLUMN in raw_df.columns else raw_df.assign(HeartDisease=np.nan))

    c1, c2 = st.columns(2)
    with c1:
        st.write("Column Names")
        st.dataframe(pd.DataFrame({"Column": raw_df.columns}), use_container_width=True)
        st.write("Data Types")
        st.dataframe(raw_df.dtypes.astype(str).reset_index().rename(columns={"index": "Column", 0: "Data Type"}))
    with c2:
        st.write("Missing Value Counts")
        st.dataframe(raw_df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing"}))
        st.metric("Duplicate Row Count", int(raw_df.duplicated().sum()))

    st.write("Statistical Summary")
    st.dataframe(raw_df.describe(include="all").T, use_container_width=True)

    if not missing:
        cleaned_df, before_after = clean_heart_data(raw_df)
        st.subheader("Cleaning Result")
        st.write(
            "RestingBP and Cholesterol values of 0 are replaced with missing values and then "
            "imputed with each column median, matching the notebook preprocessing."
        )
        st.dataframe(before_after, use_container_width=True)
        st.dataframe(cleaned_df.head(10), use_container_width=True)
        if mode == "Training data":
            uploaded_fingerprint = dataframe_fingerprint(raw_df)
            if uploaded_fingerprint != st.session_state.get("active_fingerprint"):
                set_active_training_dataset(raw_df, source)
                st.success("Uploaded training dataset applied to the full dashboard. Recomputing all pages...")
                st.rerun()
            st.success("This training dataset is active for EDA, predictive modeling, risk scoring, Apriori, and K-Means.")
        else:
            st.info(
                "Prediction-mode uploads are validated and cleaned here, but they are not used to retrain the "
                "dashboard because HeartDisease is optional. Use the New Patient Prediction page for batch predictions."
            )
        st.download_button(
            "Download cleaned dataset as CSV",
            data=dataframe_to_csv(cleaned_df),
            file_name="vitalsights_cleaned_dataset.csv",
            mime="text/csv",
        )


def render_eda(cleaned_df: pd.DataFrame, raw_df: pd.DataFrame):
    st.title("Exploratory Data Analysis")

    tabs = st.tabs(
        [
            "Quality",
            "Univariate",
            "Zero Handling",
            "Bivariate",
            "Multivariate",
        ]
    )

    with tabs[0]:
        show_notebook_markdown_cell(7)
        st.pyplot(fig_null_heatmap(cleaned_df))
        st.write("Missing Value Counts")
        st.dataframe(cleaned_df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing"}))
        st.metric("Duplicate Rows", int(cleaned_df.duplicated().sum()))
        show_notebook_markdown_cell(8)
        st.write("Unique Values for Categorical Columns")
        unique_rows = []
        for column in CATEGORICAL_COLUMNS:
            unique_rows.append({"Column": column, "Unique Values": ", ".join(map(str, cleaned_df[column].unique()))})
        st.dataframe(pd.DataFrame(unique_rows), use_container_width=True)
        show_notebook_markdown_cell(9)

    with tabs[1]:
        show_notebook_markdown_cell(11)
        st.pyplot(fig_univariate(cleaned_df))
        show_notebook_markdown_cell(12)
        st.pyplot(fig_kde_numeric(cleaned_df))
        st.pyplot(fig_box_numeric(cleaned_df))
        show_notebook_markdown_cell(13)

    with tabs[2]:
        show_notebook_markdown_cell(15)
        _, before_after = clean_heart_data(raw_df)
        st.dataframe(before_after, use_container_width=True)
        st.pyplot(fig_cleaned_distributions(cleaned_df))
        show_notebook_markdown_cell(16)

    with tabs[3]:
        show_notebook_markdown_cell(18)
        st.pyplot(fig_categorical_by_target(cleaned_df))
        show_notebook_markdown_cell(19)
        st.pyplot(fig_numeric_by_target(cleaned_df))
        show_notebook_markdown_cell(20)

    with tabs[4]:
        show_notebook_markdown_cell(22)
        st.pyplot(fig_correlation(cleaned_df))
        show_notebook_markdown_cell(23)
        st.pyplot(fig_pairplot(cleaned_df))
        show_notebook_markdown_cell(24)


def render_predictive_modeling(bundle: dict):
    st.title("Predictive Modeling")
    show_notebook_markdown_cell(30)
    show_notebook_markdown_cell(32)
    show_notebook_markdown_cell(34)
    show_notebook_markdown_cell(36)
    st.subheader("Model Comparison")
    st.caption("Metrics are recomputed from the active dashboard dataset.")
    st.dataframe(bundle["metrics_df"], use_container_width=True)
    st.pyplot(fig_confusion_matrices(bundle))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Decision Tree Classification Report")
        st.dataframe(
            classification_report_table(bundle["y_test"], bundle["y_pred_dt"]),
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.subheader("Random Forest Classification Report")
        st.dataframe(
            classification_report_table(bundle["y_test"], bundle["y_pred_rf"]),
            use_container_width=True,
            hide_index=True,
        )
    show_notebook_markdown_cell(37)


def render_interpretability(bundle: dict):
    st.title("Model Interpretability")
    show_notebook_markdown_cell(39)
    st.subheader("Decision Tree Clinical Flowchart")
    st.pyplot(fig_decision_tree(bundle["dt_model"], bundle["model_columns"]))

    st.subheader("Readable Decision Tree Rules")
    rules = export_text(bundle["dt_model"], feature_names=bundle["model_columns"])
    st.text_area("Decision Tree Rules", value=rules, height=360)
    show_notebook_markdown_cell(40)

    st.subheader("Representative Random Forest Estimator")
    st.pyplot(fig_decision_tree(bundle["rf_model"].estimators_[0], bundle["model_columns"], max_depth=3))

    st.subheader("Feature Importance")
    st.pyplot(fig_feature_importance(bundle))
    show_notebook_markdown_cell(41)
    show_notebook_markdown_cell(42)


def render_risk_stratification(bundle: dict):
    st.title("Clinical Risk Stratification")
    show_notebook_markdown_cell(44)
    risk_df = make_patient_risk_df(bundle)
    st.subheader("Patient Risk Data Preview")
    st.dataframe(risk_df.head(10), use_container_width=True)
    distribution = risk_df["ACC_AHA_Risk_Tier"].value_counts().reindex(RISK_LABELS).fillna(0).astype(int)
    st.dataframe(distribution.rename("Patient Count").reset_index().rename(columns={"index": "Risk Tier"}))

    triage_df = risk_df.sort_values(by="Heart_Disease_Probability", ascending=False).reset_index(drop=True)
    display_triage = triage_df.copy()
    display_triage["Heart_Disease_Probability"] = display_triage["Heart_Disease_Probability"].map("{:.1%}".format)
    st.subheader("Top 10 Highest-Risk Patients")
    st.dataframe(display_triage.head(10), use_container_width=True)
    st.pyplot(fig_risk_tiers(risk_df))
    show_notebook_markdown_cell(45)
    st.download_button(
        "Download triage list",
        data=dataframe_to_csv(triage_df),
        file_name="vitalsights_triage_list.csv",
        mime="text/csv",
    )


def render_apriori(cleaned_df: pd.DataFrame, apriori_bundle: dict):
    st.title("Apriori Association Rules")
    if apriori_bundle.get("error"):
        st.error(apriori_bundle["error"])
        return

    pruned = apriori_bundle["pruned_rules"]
    if pruned.empty:
        st.warning("No heart disease association rules met the configured support and lift thresholds.")
        return

    mined_preview = prepare_apriori_frame(cleaned_df)[
        ["Age_Group", "BP_Level", "Chol_Level", "MaxHR_Level", "ST_Depression", "HeartDisease_Lab"]
    ].head(10)
    st.dataframe(mined_preview, use_container_width=True)

    st.subheader("Transactional Matrix Preview")
    st.dataframe(apriori_bundle["encoded"].head(10), use_container_width=True)
    
    show_notebook_markdown_cell(58)
    display_rules = pruned[["antecedents", "support", "confidence", "lift"]].copy()
    display_rules["antecedents"] = display_rules["antecedents"].apply(format_itemset)
    st.subheader("Top Pruned Heart Disease Rules")
    st.dataframe(display_rules.head(20), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Frequent Itemsets", f"{len(apriori_bundle['frequent_itemsets']):,}")
    c2.metric("Pruned Risk Rules", f"{len(pruned):,}")
    show_notebook_markdown_cell(59)

    show_notebook_markdown_cell(60)
    st.pyplot(fig_apriori_bar(pruned))
    show_notebook_markdown_cell(61)
    st.pyplot(fig_apriori_network(pruned))
    show_notebook_markdown_cell(62)
    st.pyplot(fig_rule_strength_heatmap(pruned))
    show_notebook_markdown_cell(63)

    show_notebook_markdown_cell(64)
    st.subheader("Clinical Decision Support Rulebook")
    rulebook = apriori_bundle["rulebook"].head(25)
    st.dataframe(rulebook, use_container_width=True)
    show_notebook_markdown_cell(65)
    text_rulebook = "\n\n".join(rulebook["Plain English Rule"].tolist())
    st.download_button(
        "Download rulebook as CSV",
        data=dataframe_to_csv(apriori_bundle["rulebook"]),
        file_name="vitalsights_rulebook.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download rulebook as text",
        data=text_rulebook.encode("utf-8"),
        file_name="vitalsights_rulebook.txt",
        mime="text/plain",
    )


def render_kmeans(cluster_bundle: dict):
    st.title("K-Means Clustering")
    show_notebook_markdown_cell(66)
    show_notebook_markdown_cell(67)
    st.subheader("Scaled Data Preview")
    st.dataframe(cluster_bundle["X_scaled_df"].head(), use_container_width=True)

    show_notebook_markdown_cell(68)
    st.subheader("Elbow Method")
    st.pyplot(fig_elbow(cluster_bundle["wcss"]))
    show_notebook_markdown_cell(69)

    show_notebook_markdown_cell(70)
    show_notebook_markdown_cell(71)
    st.subheader("Patient Segmentation Map")
    st.pyplot(fig_cluster_scatter(cluster_bundle["clustered_df"]))
    show_notebook_markdown_cell(72)

    show_notebook_markdown_cell(73)
    st.subheader("Cluster Profile")
    profile = cluster_bundle["cluster_profile"].copy()
    profile["HeartDisease_Prevalence"] = profile[TARGET_COLUMN].map("{:.1%}".format)
    st.dataframe(profile, use_container_width=True)
    show_notebook_markdown_cell(74)
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download clustered patient data",
        data=dataframe_to_csv(cluster_bundle["clustered_df"]),
        file_name="vitalsights_clustered_patients.csv",
        mime="text/csv",
    )
    c2.download_button(
        "Download cluster profile",
        data=dataframe_to_csv(cluster_bundle["cluster_profile"].reset_index()),
        file_name="vitalsights_cluster_profile.csv",
        mime="text/csv",
    )


def single_patient_form() -> pd.DataFrame:
    with st.form("single_patient_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", min_value=1, max_value=120, value=54)
        sex = c2.selectbox("Sex", ["M", "F"])
        chest_pain = c3.selectbox("ChestPainType", ["ATA", "NAP", "ASY", "TA"], index=2)
        resting_bp = c1.number_input("RestingBP", min_value=0, max_value=300, value=130)
        cholesterol = c2.number_input("Cholesterol", min_value=0, max_value=700, value=240)
        fasting_bs = c3.selectbox("FastingBS", [0, 1])
        resting_ecg = c1.selectbox("RestingECG", ["Normal", "ST", "LVH"])
        max_hr = c2.number_input("MaxHR", min_value=40, max_value=220, value=140)
        exercise_angina = c3.selectbox("ExerciseAngina", ["N", "Y"])
        oldpeak = c1.number_input("Oldpeak", min_value=-5.0, max_value=10.0, value=1.0, step=0.1)
        st_slope = c2.selectbox("ST_Slope", ["Up", "Flat", "Down"], index=1)
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "Age": age,
                "Sex": sex,
                "ChestPainType": chest_pain,
                "RestingBP": resting_bp,
                "Cholesterol": cholesterol,
                "FastingBS": fasting_bs,
                "RestingECG": resting_ecg,
                "MaxHR": max_hr,
                "ExerciseAngina": exercise_angina,
                "Oldpeak": oldpeak,
                "ST_Slope": st_slope,
            }
        ]
    )


def render_prediction(cleaned_df: pd.DataFrame, model_bundle: dict, cluster_bundle: dict, apriori_bundle: dict):
    st.title("New Patient Prediction")
    mode = st.radio("Prediction mode", ["Single Patient Form", "Batch Upload"], horizontal=True)

    input_df = pd.DataFrame()
    if mode == "Single Patient Form":
        input_df = single_patient_form()
    else:
        uploaded = st.file_uploader("Upload patient CSV or Excel file", type=["csv", "xlsx", "xls"], key="prediction_upload")
        if uploaded:
            input_df = read_uploaded_table(uploaded)
            missing = validate_columns(input_df, training_mode=False)
            if missing:
                st.error(f"Missing required prediction columns: {', '.join(missing)}")
                input_df = pd.DataFrame()
            else:
                st.success("Prediction columns validated.")
                st.dataframe(input_df.head(10), use_container_width=True)

    if not input_df.empty:
        result = predict_patients(input_df, cleaned_df, model_bundle, cluster_bundle, apriori_bundle)
        st.subheader("Prediction Results")
        display = result.copy()
        display["Heart_Disease_Probability"] = display["Heart_Disease_Probability"].map("{:.1%}".format)
        st.dataframe(display, use_container_width=True)

        if len(result) == 1:
            row = result.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prediction", row["Predicted_Label"])
            c2.metric("Probability", f"{row['Heart_Disease_Probability_Percent']:.1f}%")
            c3.metric("Risk Tier", row["ACC_AHA_Risk_Tier"])
            c4.metric("Cluster", int(row["Clinical_Cluster"]))
            st.write(f"Clinical action: {row['Clinical_Action']}")
            st.write(f"Cluster interpretation: {row['Cluster_Risk_Interpretation']}")
            if "Matched_Apriori_Rules" in row:
                st.write(f"Matched descriptive rules: {row['Matched_Apriori_Rules']}")

        st.download_button(
            "Download prediction results",
            data=dataframe_to_csv(result),
            file_name="vitalsights_predictions.csv",
            mime="text/csv",
        )


def main():
    initialize_active_dataset()
    raw_df = st.session_state["active_raw_df"]
    cleaned_df = st.session_state["active_cleaned_df"]
    active_source = st.session_state["active_source"]
    model_bundle = train_predictive_models(cleaned_df)
    cluster_bundle = train_kmeans(cleaned_df)
    apriori_bundle = run_apriori(cleaned_df)

    if "page" not in st.session_state:
        st.session_state["page"] = PAGES[0]
    default_page_index = PAGES.index(st.session_state["page"])
    page = st.sidebar.selectbox(
        "VitalSights Pages",
        PAGES,
        index=default_page_index,
    )
    st.session_state["page"] = page
    st.sidebar.caption(f"Active dataset: {active_source}")
    st.sidebar.caption("Clinical decision-support prototype for data mining coursework.")

    if page == "Home / Project Overview":
        render_home(cleaned_df, active_source)
    elif page == "Data Upload and Cleaning":
        render_upload_cleaning()
    elif page == "Exploratory Data Analysis":
        render_eda(cleaned_df, raw_df)
    elif page == "Predictive Modeling":
        render_predictive_modeling(model_bundle)
    elif page == "Model Interpretability":
        render_interpretability(model_bundle)
    elif page == "Clinical Risk Stratification":
        render_risk_stratification(model_bundle)
    elif page == "Apriori Association Rules":
        render_apriori(cleaned_df, apriori_bundle)
    elif page == "K-Means Clustering":
        render_kmeans(cluster_bundle)
    elif page == "New Patient Prediction":
        render_prediction(cleaned_df, model_bundle, cluster_bundle, apriori_bundle)


if __name__ == "__main__":
    main()

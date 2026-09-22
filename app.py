"""
Airline Passenger Satisfaction Analytics & ML Dashboard
========================================================
Single-file Streamlit application covering:
  - Data loading & preprocessing
  - Exploratory Data Analysis (EDA)
  - ML model training & evaluation (Logistic Regression, Random Forest, XGBoost)
  - Interactive dashboard with KPIs, filters, and prediction

Run:  python -m streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "airline_satisfaction_data.csv"
MODEL_DIR = "models"
PALETTE = {"satisfied": "#3b82d4", "neutral or dissatisfied": "#e74c3c"}
RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset …")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Fix typo in Class column
    df["Class"] = df["Class"].replace("Busi", "Business")

    # Standardise Customer Type casing
    df["Customer Type"] = df["Customer Type"].str.strip().str.title()

    # Convert boolean satisfaction → string label
    df["satisfaction"] = df["satisfaction"].map(
        {True: "satisfied", False: "neutral or dissatisfied"}
    )

    # Drop rows where satisfaction is still NaN after mapping
    df = df.dropna(subset=["satisfaction"])

    # Handle Arrival Delay NaNs (fill with median)
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(
        df["Arrival Delay in Minutes"].median()
    )

    # Clip rating columns to valid range 0-5
    for col in RATING_COLS:
        df[col] = df[col].clip(0, 5)

    # Age bins
    bins = [0, 18, 30, 45, 60, 100]
    labels = ["<18", "18-30", "31-45", "46-60", "60+"]
    df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

    # Flight Distance bins
    dist_bins = [0, 500, 1500, 3000, 5000]
    dist_labels = ["Short (<500)", "Medium (500-1500)", "Long (1500-3000)", "Very Long (3000+)"]
    df["Distance Category"] = pd.cut(df["Flight Distance"], bins=dist_bins, labels=dist_labels)

    return df


@st.cache_data(show_spinner="Preprocessing for ML …")
def preprocess_for_ml(df: pd.DataFrame):
    """Return X, y, feature names, scaler, encoders for model training."""
    ml_df = df.copy()

    cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        encoders[col] = le

    feature_cols = cat_cols + ["Age", "Flight Distance"] + RATING_COLS + [
        "Departure Delay in Minutes", "Arrival Delay in Minutes"
    ]

    X = ml_df[feature_cols].values
    y = (ml_df["satisfaction"] == "satisfied").astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols, scaler, encoders


@st.cache_resource(show_spinner="Training models … (first run only)")
def train_models(df: pd.DataFrame):
    """Train LR, RF, GBM models and return results."""
    X, y, feature_cols, scaler, encoders = preprocess_for_ml(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12,
                                                random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150,
                                                         learning_rate=0.1,
                                                         max_depth=5,
                                                         random_state=SEED),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "cm": confusion_matrix(y_test, y_pred),
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc(fpr, tpr),
        }

    return results, feature_cols, scaler, encoders, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return buf


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.edgecolor": "#dee2e6",
        "axes.grid": True,
        "grid.color": "#e9ecef",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "font.family": "sans-serif",
        "font.size": 11,
    })


def plot_satisfaction_distribution(df):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df["satisfaction"].value_counts()
    colors = [PALETTE["satisfied"], PALETTE["neutral or dissatisfied"]]
    axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=colors, startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0].set_title("Satisfaction Distribution", fontweight="bold")

    bars = axes[1].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Passenger Count by Satisfaction", fontweight="bold")
    axes[1].set_xlabel("Satisfaction")
    axes[1].set_ylabel("Count")
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f"{bar.get_height():,}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    return fig


def plot_demographic_breakdown(df):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cat_cols = [("Gender", axes[0, 0]), ("Customer Type", axes[0, 1]),
                ("Type of Travel", axes[1, 0]), ("Class", axes[1, 1])]
    for col, ax in cat_cols:
        ct = df.groupby([col, "satisfaction"]).size().unstack(fill_value=0)
        ct.plot(kind="bar", ax=ax, color=[PALETTE["satisfied"],
                                           PALETTE["neutral or dissatisfied"]],
                edgecolor="white", linewidth=1.2)
        ax.set_title(f"Satisfaction by {col}", fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Satisfaction", fontsize=9)
    plt.tight_layout()
    return fig


def plot_age_distribution(df):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sat_colors = [PALETTE["satisfied"], PALETTE["neutral or dissatisfied"]]

    for sat_label, color in zip(["satisfied", "neutral or dissatisfied"], sat_colors):
        subset = df[df["satisfaction"] == sat_label]["Age"]
        axes[0].hist(subset, bins=30, alpha=0.65, label=sat_label, color=color, edgecolor="white")
    axes[0].set_title("Age Distribution by Satisfaction", fontweight="bold")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    age_sat = df.groupby(["Age Group", "satisfaction"]).size().unstack(fill_value=0)
    age_sat.plot(kind="bar", ax=axes[1],
                 color=[PALETTE["satisfied"], PALETTE["neutral or dissatisfied"]],
                 edgecolor="white")
    axes[1].set_title("Satisfaction by Age Group", fontweight="bold")
    axes[1].set_xlabel("Age Group")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    return fig


def plot_flight_distance(df):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for sat_label, color in zip(["satisfied", "neutral or dissatisfied"],
                                  [PALETTE["satisfied"], PALETTE["neutral or dissatisfied"]]):
        subset = df[df["satisfaction"] == sat_label]["Flight Distance"]
        axes[0].hist(subset, bins=40, alpha=0.65, label=sat_label, color=color, edgecolor="white")
    axes[0].set_title("Flight Distance Distribution by Satisfaction", fontweight="bold")
    axes[0].set_xlabel("Flight Distance (km)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    dist_sat = df.groupby(["Distance Category", "satisfaction"]).size().unstack(fill_value=0)
    dist_sat.plot(kind="bar", ax=axes[1],
                  color=[PALETTE["satisfied"], PALETTE["neutral or dissatisfied"]],
                  edgecolor="white")
    axes[1].set_title("Satisfaction by Distance Category", fontweight="bold")
    axes[1].set_xlabel("Distance Category")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    return fig


def plot_ratings_heatmap(df):
    set_style()
    avg_ratings = df.groupby("satisfaction")[RATING_COLS].mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(avg_ratings, annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, vmin=1, vmax=5,
                cbar_kws={"label": "Avg Rating (1-5)"})
    ax.set_title("Average Service Ratings by Satisfaction", fontweight="bold", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Satisfaction Level")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    plt.tight_layout()
    return fig


def plot_ratings_radar(df):
    set_style()
    avg_ratings = df.groupby("satisfaction")[RATING_COLS].mean()
    labels = [c.replace(" ", "\n") for c in RATING_COLS]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for sat_label, color in zip(["satisfied", "neutral or dissatisfied"],
                                  [PALETTE["satisfied"], PALETTE["neutral or dissatisfied"]]):
        values = avg_ratings.loc[sat_label].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=sat_label)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], size=8)
    ax.set_title("Service Ratings Radar Chart", fontweight="bold", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df):
    set_style()
    num_cols = ["Age", "Flight Distance"] + RATING_COLS + [
        "Departure Delay in Minutes", "Arrival Delay in Minutes"
    ]
    corr_df = df[num_cols].copy()
    corr_df["satisfaction_num"] = (df["satisfaction"] == "satisfied").astype(int)
    corr = corr_df.corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=13)
    plt.tight_layout()
    return fig


def plot_delay_analysis(df):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    delay_cols = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
    for i, col in enumerate(delay_cols):
        for sat_label, color in zip(["satisfied", "neutral or dissatisfied"],
                                     [PALETTE["satisfied"], PALETTE["neutral or dissatisfied"]]):
            subset = df[df["satisfaction"] == sat_label][col].clip(0, 120)
            axes[i].hist(subset, bins=40, alpha=0.65, label=sat_label,
                         color=color, edgecolor="white")
        axes[i].set_title(f"{col} Distribution (capped at 120)", fontweight="bold")
        axes[i].set_xlabel(f"{col}")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, model_name):
    set_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Dissatisfied", "Satisfied"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor="#dee2e6")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    return fig


def plot_roc_curves(results):
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3b82d4", "#27ae60", "#e67e22"]
    for (name, res), color in zip(results.items(), colors):
        ax.plot(res["fpr"], res["tpr"], color=color, linewidth=2,
                label=f"{name} (AUC = {res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_title("ROC Curves — All Models", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_feature_importance(results, feature_cols):
    set_style()
    rf_model = results["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh([feature_cols[i] for i in idx[::-1]],
                   importances[idx[::-1]],
                   color="#3b82d4", edgecolor="white")
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance Score")
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=9)
    plt.tight_layout()
    return fig


def plot_model_comparison(results):
    set_style()
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.22
    colors = ["#3b82d4", "#27ae60", "#e67e22"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color,
                      edgecolor="white", linewidth=1.2)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def predict_satisfaction(input_dict, model, scaler, encoders, feature_cols):
    row = {}
    cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
    for col in cat_cols:
        le = encoders[col]
        val = input_dict[col]
        # Handle unseen labels gracefully
        if val in le.classes_:
            row[col] = le.transform([val])[0]
        else:
            row[col] = 0
    for col in feature_cols:
        if col not in cat_cols:
            row[col] = input_dict.get(col, 0)

    X = np.array([[row[c] for c in feature_cols]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    return pred, prob


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Airline Satisfaction Analytics",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ──────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
        border-left: 4px solid #3b82d4;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .metric-card h4 { color: #57606a; font-size: 13px; margin: 0 0 4px 0; }
    .metric-card h2 { color: #1f2328; font-size: 26px; margin: 0; }
    .section-header {
        background: #3b82d4;
        color: white;
        padding: 10px 18px;
        border-radius: 6px;
        font-size: 18px;
        font-weight: 700;
        margin: 20px 0 14px 0;
    }
    .predict-box {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────
    df_full = load_data(DATA_PATH)

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Airplane_silhouette.svg/120px-Airplane_silhouette.svg.png",
                     width=80)
    st.sidebar.title("✈️ Airline Satisfaction")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview & KPIs", "🔍 EDA & Visualizations",
         "🤖 ML Model Performance", "🎯 Predict Satisfaction"],
        index=0,
    )

    st.sidebar.markdown("### 🔧 Filters")
    gender_opts = ["All"] + sorted(df_full["Gender"].unique().tolist())
    sel_gender = st.sidebar.selectbox("Gender", gender_opts)

    class_opts = ["All"] + sorted(df_full["Class"].unique().tolist())
    sel_class = st.sidebar.selectbox("Class", class_opts)

    travel_opts = ["All"] + sorted(df_full["Type of Travel"].unique().tolist())
    sel_travel = st.sidebar.selectbox("Type of Travel", travel_opts)

    cust_opts = ["All"] + sorted(df_full["Customer Type"].unique().tolist())
    sel_cust = st.sidebar.selectbox("Customer Type", cust_opts)

    age_min, age_max = int(df_full["Age"].min()), int(df_full["Age"].max())
    sel_age = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

    # Apply filters
    df = df_full.copy()
    if sel_gender != "All":
        df = df[df["Gender"] == sel_gender]
    if sel_class != "All":
        df = df[df["Class"] == sel_class]
    if sel_travel != "All":
        df = df[df["Type of Travel"] == sel_travel]
    if sel_cust != "All":
        df = df[df["Customer Type"] == sel_cust]
    df = df[(df["Age"] >= sel_age[0]) & (df["Age"] <= sel_age[1])]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered rows:** {len(df):,} / {len(df_full):,}")

    # ── Train models (on full dataset, cached) ───────────────────────────
    results, feature_cols, scaler, encoders, X_test, y_test = train_models(df_full)

    # ────────────────────────────────────────────────────────────────────
    # PAGE: Overview & KPIs
    # ────────────────────────────────────────────────────────────────────
    if page == "📊 Overview & KPIs":
        st.markdown("# ✈️ Airline Passenger Satisfaction Dashboard")
        st.markdown("**Interactive analytics platform for passenger satisfaction data.**")

        # KPI row
        total = len(df)
        sat_count = (df["satisfaction"] == "satisfied").sum()
        sat_pct = sat_count / total * 100 if total > 0 else 0
        avg_age = df["Age"].mean()
        avg_dist = df["Flight Distance"].mean()
        avg_delay = df["Arrival Delay in Minutes"].mean()
        loyal_pct = (df["Customer Type"] == "Loyal Customer").mean() * 100

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.markdown(f"""<div class="metric-card"><h4>Total Passengers</h4>
            <h2>{total:,}</h2></div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="metric-card"><h4>Satisfied</h4>
            <h2>{sat_count:,}</h2></div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="metric-card"><h4>Satisfaction Rate</h4>
            <h2>{sat_pct:.1f}%</h2></div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="metric-card"><h4>Avg Age</h4>
            <h2>{avg_age:.1f}</h2></div>""", unsafe_allow_html=True)
        with k5:
            st.markdown(f"""<div class="metric-card"><h4>Avg Flight Distance</h4>
            <h2>{avg_dist:.0f} km</h2></div>""", unsafe_allow_html=True)
        with k6:
            st.markdown(f"""<div class="metric-card"><h4>Avg Arrival Delay</h4>
            <h2>{avg_delay:.1f} min</h2></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Satisfaction distribution
        st.markdown('<div class="section-header">Satisfaction Distribution</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            counts = df["satisfaction"].value_counts()
            st.dataframe(
                counts.reset_index().rename(columns={"index": "Satisfaction", "satisfaction": "Count"}),
                use_container_width=True, hide_index=True
            )
            st.markdown(f"**Satisfaction rate:** {sat_pct:.2f}%")
        with c2:
            fig = plot_satisfaction_distribution(df)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Demographic breakdown
        st.markdown('<div class="section-header">Demographic Breakdown</div>', unsafe_allow_html=True)
        fig = plot_demographic_breakdown(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Quick stats table
        st.markdown('<div class="section-header">Quick Statistics</div>', unsafe_allow_html=True)
        stats_df = df.groupby("satisfaction").agg(
            Count=("Age", "count"),
            Avg_Age=("Age", "mean"),
            Avg_Distance=("Flight Distance", "mean"),
            Avg_Departure_Delay=("Departure Delay in Minutes", "mean"),
            Avg_Arrival_Delay=("Arrival Delay in Minutes", "mean"),
        ).round(2).reset_index()
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ────────────────────────────────────────────────────────────────────
    # PAGE: EDA & Visualizations
    # ────────────────────────────────────────────────────────────────────
    elif page == "🔍 EDA & Visualizations":
        st.markdown("# 🔍 Exploratory Data Analysis")

        st.markdown('<div class="section-header">Age Analysis</div>', unsafe_allow_html=True)
        fig = plot_age_distribution(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<div class="section-header">Flight Distance Analysis</div>', unsafe_allow_html=True)
        fig = plot_flight_distance(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<div class="section-header">Delay Analysis</div>', unsafe_allow_html=True)
        fig = plot_delay_analysis(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<div class="section-header">Average Service Ratings Heatmap</div>', unsafe_allow_html=True)
        fig = plot_ratings_heatmap(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<div class="section-header">Service Ratings Radar Chart</div>', unsafe_allow_html=True)
        fig = plot_ratings_radar(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
        fig = plot_correlation_matrix(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<div class="section-header">Rating Distributions per Service</div>', unsafe_allow_html=True)
        selected_rating = st.selectbox("Select a service rating to inspect", RATING_COLS)
        set_style()
        fig, ax = plt.subplots(figsize=(10, 4))
        for sat_label, color in zip(["satisfied", "neutral or dissatisfied"],
                                     [PALETTE["satisfied"], PALETTE["neutral or dissatisfied"]]):
            subset = df[df["satisfaction"] == sat_label][selected_rating]
            ax.hist(subset, bins=6, alpha=0.7, label=sat_label, color=color, edgecolor="white")
        ax.set_title(f"{selected_rating} Distribution by Satisfaction", fontweight="bold")
        ax.set_xlabel("Rating (0–5)")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<div class="section-header">Raw Data Sample</div>', unsafe_allow_html=True)
        st.dataframe(df.sample(min(100, len(df)), random_state=SEED).reset_index(drop=True),
                     use_container_width=True)

    # ────────────────────────────────────────────────────────────────────
    # PAGE: ML Model Performance
    # ────────────────────────────────────────────────────────────────────
    elif page == "🤖 ML Model Performance":
        st.markdown("# 🤖 Machine Learning Model Performance")
        st.info("Models are trained on the **full dataset** (80% train / 20% test, stratified).")

        # Summary metrics table
        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
        metrics_rows = []
        for name, res in results.items():
            metrics_rows.append({
                "Model": name,
                "Accuracy": f"{res['accuracy']:.4f}",
                "Precision": f"{res['precision']:.4f}",
                "Recall": f"{res['recall']:.4f}",
                "F1-Score": f"{res['f1']:.4f}",
                "ROC-AUC": f"{res['auc']:.4f}",
            })
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

        fig = plot_model_comparison(results)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ROC curves
        st.markdown('<div class="section-header">ROC Curves</div>', unsafe_allow_html=True)
        fig = plot_roc_curves(results)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Confusion matrices
        st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (name, res) in enumerate(results.items()):
            with cols[i]:
                fig = plot_confusion_matrix(res["cm"], name)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # Feature Importance
        st.markdown('<div class="section-header">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        fig = plot_feature_importance(results, feature_cols)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Classification report
        st.markdown('<div class="section-header">Detailed Classification Report</div>', unsafe_allow_html=True)
        model_choice = st.selectbox("Select model for detailed report", list(results.keys()))
        res = results[model_choice]
        report = classification_report(res["y_test"], res["y_pred"],
                                       target_names=["Dissatisfied", "Satisfied"])
        st.code(report, language="text")

    # ────────────────────────────────────────────────────────────────────
    # PAGE: Predict Satisfaction
    # ────────────────────────────────────────────────────────────────────
    elif page == "🎯 Predict Satisfaction":
        st.markdown("# 🎯 Predict Passenger Satisfaction")
        st.markdown("Fill in the passenger details below to predict satisfaction.")

        best_model_name = max(results, key=lambda k: results[k]["f1"])
        model_choice = st.selectbox(
            "Select Prediction Model",
            list(results.keys()),
            index=list(results.keys()).index(best_model_name),
        )

        st.markdown('<div class="predict-box">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Personal Info**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.slider("Age", 7, 85, 35)
            cust_type = st.selectbox("Customer Type",
                                      ["Loyal Customer", "Disloyal Customer"])

        with col2:
            st.markdown("**Flight Details**")
            travel_type = st.selectbox("Type of Travel",
                                        ["Business travel", "Personal Travel"])
            flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
            flight_dist = st.number_input("Flight Distance (km)", 31, 4983, 800)
            dep_delay = st.number_input("Departure Delay (min)", 0, 1000, 0)
            arr_delay = st.number_input("Arrival Delay (min)", 0, 1000, 0)

        with col3:
            st.markdown("**Service Ratings (0–5)**")
            ratings = {}
            for rname in RATING_COLS[:7]:
                ratings[rname] = st.slider(rname, 0, 5, 3, key=f"r_{rname}")

        col4, _ = st.columns([1, 2])
        with col4:
            st.markdown("**More Service Ratings (0–5)**")
            for rname in RATING_COLS[7:]:
                ratings[rname] = st.slider(rname, 0, 5, 3, key=f"r_{rname}")

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 Predict Satisfaction", type="primary"):
            input_dict = {
                "Gender": gender,
                "Customer Type": cust_type,
                "Type of Travel": travel_type,
                "Class": flight_class,
                "Age": age,
                "Flight Distance": float(flight_dist),
                "Departure Delay in Minutes": float(dep_delay),
                "Arrival Delay in Minutes": float(arr_delay),
            }
            input_dict.update({k: float(v) for k, v in ratings.items()})

            chosen_model = results[model_choice]["model"]
            pred, prob = predict_satisfaction(
                input_dict, chosen_model, scaler, encoders, feature_cols
            )

            st.markdown("---")
            pred_label = "✅ Satisfied" if pred == 1 else "❌ Neutral / Dissatisfied"
            conf = prob[1] if pred == 1 else prob[0]

            c1, c2 = st.columns(2)
            with c1:
                if pred == 1:
                    st.success(f"## Prediction: {pred_label}")
                else:
                    st.error(f"## Prediction: {pred_label}")
            with c2:
                st.metric("Confidence", f"{conf * 100:.1f}%")
                st.metric("P(Satisfied)", f"{prob[1] * 100:.1f}%")
                st.metric("P(Dissatisfied)", f"{prob[0] * 100:.1f}%")

            # Probability gauge
            set_style()
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(["Satisfied", "Dissatisfied"], [prob[1], prob[0]],
                    color=[PALETTE["satisfied"], PALETTE["neutral or dissatisfied"]])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_title("Prediction Probabilities", fontweight="bold")
            for i, v in enumerate([prob[1], prob[0]]):
                ax.text(v + 0.01, i, f"{v:.1%}", va="center")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Footer
    st.markdown("---")
    st.markdown(
        "<center><small>Airline Satisfaction Analytics Dashboard · Built with Streamlit · "
        "Data: 58,313 passengers</small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

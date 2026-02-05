import io
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


SUPPORTED_EXTENSIONS = {
    "csv": "CSV",
    "xlsx": "Excel",
    "xls": "Excel",
    "parquet": "Parquet",
}


@st.cache_data(show_spinner=False)
def load_dataframe(uploaded_file: io.BytesIO) -> pd.DataFrame:
    name = uploaded_file.name
    extension = name.split(".")[-1].lower()
    if extension == "csv":
        return pd.read_csv(uploaded_file)
    if extension in {"xlsx", "xls"}:
        return pd.read_excel(uploaded_file)
    if extension == "parquet":
        return pd.read_parquet(uploaded_file)
    raise ValueError("Unsupported file type")


def dataframe_overview(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing = dataframe.isna().sum()
    missing_pct = (missing / len(dataframe)).round(4) * 100
    unique = dataframe.nunique(dropna=True)
    return pd.DataFrame(
        {
            "dtype": dataframe.dtypes.astype(str),
            "missing": missing,
            "missing_%": missing_pct,
            "unique_values": unique,
        }
    ).sort_values(by="missing_%", ascending=False)


def split_columns(dataframe: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = dataframe.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    datetime_cols = dataframe.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    return numeric_cols, categorical_cols, datetime_cols


def detect_target_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        unique_count = series.nunique(dropna=True)
        if unique_count <= 20 and unique_count / len(series) < 0.05:
            return "classification"
        return "regression"
    return "classification"


def model_recommendations(
    dataframe: pd.DataFrame, target: str
) -> Tuple[str, List[str], List[str]]:
    target_series = dataframe[target]
    problem_type = detect_target_type(target_series)

    numeric_cols, categorical_cols, datetime_cols = split_columns(dataframe)
    text_like = [
        col
        for col in categorical_cols
        if dataframe[col].astype(str).str.len().mean() > 30
    ]

    suggestions = []
    reasons = []

    if problem_type == "classification":
        suggestions.extend(
            [
                "Logistic Regression",
                "Random Forest",
                "Gradient Boosting (XGBoost/LightGBM)",
                "Support Vector Machine",
            ]
        )
        reasons.append("Target looks categorical or has few discrete values.")
    else:
        suggestions.extend(
            [
                "Linear Regression / ElasticNet",
                "Random Forest Regressor",
                "Gradient Boosting Regressor (XGBoost/LightGBM)",
                "Support Vector Regression",
            ]
        )
        reasons.append("Target is numeric with many unique values.")

    if datetime_cols:
        suggestions.append("Time-series models (ARIMA/Prophet) if target depends on time")
        reasons.append("Datetime columns detected; consider temporal modeling.")

    if text_like:
        suggestions.append("TF-IDF + Linear Model or Naive Bayes for text features")
        reasons.append("Text-like columns detected; add NLP pipeline for those features.")

    if len(dataframe) > 100_000:
        reasons.append("Large dataset detected; tree ensembles and linear models scale well.")
    elif len(dataframe) < 1000:
        reasons.append("Small dataset; start with simpler, regularized models.")

    preprocessing_tips = []
    if categorical_cols:
        preprocessing_tips.append("Encode categorical features (one-hot, target encoding).")
    if numeric_cols:
        preprocessing_tips.append("Scale numeric features for distance-based models.")
    if dataframe.isna().any().any():
        preprocessing_tips.append("Handle missing values (imputation or drop).")

    return problem_type, suggestions, reasons + preprocessing_tips


def render_trend_plot(dataframe: pd.DataFrame, numeric_cols: List[str]) -> None:
    st.subheader("Trend explorer")
    if not numeric_cols:
        st.info("Add numeric columns to see trend plots.")
        return

    columns = dataframe.columns.tolist()
    x_axis = st.selectbox("X axis", columns)
    y_axis = st.selectbox("Y axis (numeric)", numeric_cols)

    if x_axis and y_axis:
        fig, ax = plt.subplots()
        plot_df = dataframe[[x_axis, y_axis]].dropna()
        if plot_df.empty:
            st.warning("No data available for the selected axes.")
            return
        ax.plot(plot_df[x_axis], plot_df[y_axis], marker="o", linestyle="-")
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{y_axis} over {x_axis}")
        st.pyplot(fig)


def render_eda(dataframe: pd.DataFrame) -> None:
    st.subheader("Data overview")
    st.write(f"Rows: {len(dataframe):,} | Columns: {dataframe.shape[1]}")
    st.dataframe(dataframe.head(25))

    overview = dataframe_overview(dataframe)
    st.subheader("Column summary")
    st.dataframe(overview)

    numeric_cols, categorical_cols, datetime_cols = split_columns(dataframe)

    st.subheader("Summary statistics")
    if numeric_cols:
        st.dataframe(dataframe[numeric_cols].describe().T)
    if categorical_cols:
        st.dataframe(dataframe[categorical_cols].describe().T)

    if numeric_cols:
        st.subheader("Correlation heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            dataframe[numeric_cols].corr(),
            ax=ax,
            cmap="coolwarm",
            annot=False,
            linewidths=0.5,
        )
        st.pyplot(fig)

    st.subheader("Distribution explorer")
    if numeric_cols:
        numeric_col = st.selectbox("Numeric column", numeric_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(dataframe[numeric_col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {numeric_col}")
        st.pyplot(fig)

    if categorical_cols:
        categorical_col = st.selectbox("Categorical column", categorical_cols)
        value_counts = dataframe[categorical_col].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax)
        ax.set_title(f"Top categories for {categorical_col}")
        ax.set_xlabel("Count")
        ax.set_ylabel(categorical_col)
        st.pyplot(fig)

    if datetime_cols:
        st.subheader("Datetime columns detected")
        st.write(
            "Datetime columns can drive time-based trends. Consider resampling for hourly/daily insights."
        )

    render_trend_plot(dataframe, numeric_cols)


def render_model_suggestions(dataframe: pd.DataFrame) -> None:
    st.subheader("Model suggestion")
    target = st.selectbox("Select target column", dataframe.columns)

    if not target:
        st.info("Select a target column to see model suggestions.")
        return

    problem_type, suggestions, reasons = model_recommendations(dataframe, target)

    st.write(f"**Detected problem type:** {problem_type.title()}")

    st.markdown("**Recommended models**")
    for model in suggestions:
        st.write(f"- {model}")

    st.markdown("**Why these models?**")
    for reason in reasons:
        st.write(f"- {reason}")


def main() -> None:
    st.set_page_config(page_title="Smart EDA Analyzer", layout="wide")
    st.title("Smart EDA Analyzer")
    st.write(
        "Upload a dataset to explore key trends and receive model recommendations tailored to your target."
    )

    uploaded_file = st.file_uploader(
        "Upload a dataset", type=list(SUPPORTED_EXTENSIONS.keys())
    )

    if not uploaded_file:
        st.info("Upload a CSV, Excel, or Parquet file to get started.")
        return

    with st.spinner("Loading data..."):
        dataframe = load_dataframe(uploaded_file)

    st.success(
        f"Loaded {uploaded_file.name} ({SUPPORTED_EXTENSIONS[uploaded_file.name.split('.')[-1].lower()]})"
    )

    tab_eda, tab_model = st.tabs(["EDA", "Model Suggestion"])

    with tab_eda:
        render_eda(dataframe)

    with tab_model:
        render_model_suggestions(dataframe)


if __name__ == "__main__":
    main()

# ============================================================
# PROFESSIONAL PREDICTIVE ANALYTICS DASHBOARD (FINAL FIXED)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import scipy.stats as stats
import pymannkendall as mk

# PDF Libraries
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

import os

st.set_page_config(layout="wide")
st.title("Predictive Analytics Professional Dashboard")

# ================= SESSION =================

if "df" not in st.session_state:
    st.session_state.df = None

if "results" not in st.session_state:
    st.session_state.results = []

if "outliers" not in st.session_state:
    st.session_state.outliers = None

# ================= FILE UPLOAD =================

file = st.file_uploader("Upload CSV File")

if file:
    st.session_state.df = pd.read_csv(file)
    st.success("Dataset Loaded")

df = st.session_state.df

# ============================================================
# MODULES
# ============================================================

if df is not None:

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    module = st.sidebar.selectbox(
        "Select Module",
        [
            "Outlier Detection",
            "Outlier Handling",
            "Scaling",
            "Linear Regression",
            "Theil-Sen Regression"
        ]
    )

# ============================================================
# OUTLIER DETECTION
# ============================================================

    if module == "Outlier Detection":

        col = st.sidebar.selectbox("Select Column", numeric_cols)

        if st.sidebar.button("Detect Outliers"):

            z = np.abs(stats.zscore(df[col]))
            outliers = df[z > 3]

            st.session_state.outliers = outliers

            st.write("Outliers Found:", len(outliers))
            st.dataframe(outliers)

# ============================================================
# OUTLIER HANDLING (FIXED)
# ============================================================

    if module == "Outlier Handling":

        method = st.sidebar.selectbox(
            "Handling Method",
            ["Remove", "Cap"]
        )

        if st.sidebar.button("Apply Outlier Handling"):

            if st.session_state.outliers is None:
                st.warning("Run Outlier Detection First")

            else:

                df_copy = df.copy()

                if method == "Remove":
                    df_copy = df_copy.drop(st.session_state.outliers.index)

                else:  # Cap
                    for col in numeric_cols:
                        lower = df_copy[col].quantile(0.01)
                        upper = df_copy[col].quantile(0.99)
                        df_copy[col] = np.clip(df_copy[col], lower, upper)

                st.session_state.df = df_copy

                st.success("Outliers Handled Successfully")
                st.dataframe(df_copy.head())

# ============================================================
# SCALING (FIXED)
# ============================================================

    if module == "Scaling":

        scaler_type = st.sidebar.selectbox(
            "Select Scaler",
            ["Standard", "MinMax", "Robust"]
        )

        if st.sidebar.button("Apply Scaling"):

            df_copy = df.copy()

            if scaler_type == "Standard":
                scaler = StandardScaler()

            elif scaler_type == "MinMax":
                scaler = MinMaxScaler()

            else:
                scaler = RobustScaler()

            df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])

            st.session_state.df = df_copy

            st.success("Scaling Applied")
            st.dataframe(df_copy.head())

# ============================================================
# LINEAR REGRESSION
# ============================================================

    if module == "Linear Regression":

        target = st.sidebar.selectbox("Target", numeric_cols)

        features = st.sidebar.multiselect(
            "Independent Variables",
            [col for col in numeric_cols if col != target]
        )

        if st.sidebar.button("Run Linear Regression"):

            if len(features) == 0:
                st.warning("Select Features")

            else:
                X = df[features]
                y = df[target]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = LinearRegression()
                model.fit(X_train, y_train)

                pred = model.predict(X_test)

                r2 = r2_score(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))

                st.write("R²:", r2)
                st.write("RMSE:", rmse)

# ============================================================
# THEIL-SEN REGRESSION (FIXED)
# ============================================================

    if module == "Theil-Sen Regression":

        target = st.sidebar.selectbox("Target Variable", numeric_cols)

        feature = st.sidebar.selectbox(
            "Independent Variable",
            [col for col in numeric_cols if col != target]
        )

        if st.sidebar.button("Run Theil-Sen Regression"):

            X = df[[feature]]
            y = df[target]

            model = TheilSenRegressor()
            model.fit(X, y)

            st.write("Coefficient:", model.coef_[0])
            st.write("Intercept:", model.intercept_)

# ============================================================
# PDF DOWNLOAD (UPDATED)
# ============================================================

    def generate_pdf():

        file_path = "analysis_report.pdf"
        doc = SimpleDocTemplate(file_path)
        elements = []

        styles = getSampleStyleSheet()
        style = styles["Normal"]

        elements.append(Paragraph("Predictive Analytics Report", styles["Heading1"]))
        elements.append(Spacer(1, 0.5 * inch))

        for result in st.session_state.results:
            elements.append(Paragraph(str(result), style))
            elements.append(Spacer(1, 0.2 * inch))

        doc.build(elements)

        return file_path


    if st.sidebar.button("Download Results as PDF"):

        pdf_path = generate_pdf()

        with open(pdf_path, "rb") as f:
            st.sidebar.download_button(
                "Click to Download PDF",
                f,
                file_name="analysis_report.pdf"
            )

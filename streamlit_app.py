import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import scipy.stats as stats
import pymannkendall as mk


# ================================
# PAGE CONFIG
# ================================

st.set_page_config(layout="wide")
st.title("Predictive Analytics Professional Dashboard")


# ================================
# SESSION STATE INITIALIZATION
# ================================

if "df" not in st.session_state:
    st.session_state.df = None

if "results" not in st.session_state:
    st.session_state.results = []

if "outliers" not in st.session_state:
    st.session_state.outliers = None


# ================================
# FILE UPLOAD
# ================================

file = st.file_uploader("Upload CSV")

if file:
    st.session_state.df = pd.read_csv(file)
    st.success("Data loaded successfully")


df = st.session_state.df


# ================================
# MAIN LOGIC
# ================================

if df is not None:

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.sidebar.title("Control Panel")

    module = st.sidebar.selectbox(
        "Select Module",
        [
            "EDA",
            "Missing Value Handling",
            "Outlier Detection",
            "Outlier Handling",
            "Scaling",
            "Linear Regression",
            "Theil-Sen Regression",
            "Mann-Kendall Test",
            "PCA"
        ]
    )

    # ================================
    # EDA
    # ================================

    if module == "EDA":

        if st.sidebar.button("Run EDA"):

            result = {
                "title": "EDA Results",
                "data": df.describe()
            }

            st.session_state.results.append(result)

    # ================================
    # MISSING VALUE HANDLING
    # ================================

    if module == "Missing Value Handling":

        method = st.sidebar.selectbox(
            "Method",
            ["Mean", "Median", "KNN"]
        )

        if st.sidebar.button("Apply Imputation"):

            df_copy = df.copy()

            if method == "Mean":

                imputer = SimpleImputer(strategy="mean")
                df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])

            elif method == "Median":

                imputer = SimpleImputer(strategy="median")
                df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])

            elif method == "KNN":

                imputer = KNNImputer()
                df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])

            st.session_state.df = df_copy

            st.session_state.results.append({
                "title": f"Imputation Applied ({method})",
                "data": df_copy.head()
            })


    # ================================
    # OUTLIER DETECTION
    # ================================

    if module == "Outlier Detection":

        col = st.sidebar.selectbox("Column", numeric_cols)

        method = st.sidebar.selectbox(
            "Method",
            ["ZScore", "IQR", "Isolation Forest", "LOF"]
        )

        if st.sidebar.button("Detect Outliers"):

            if method == "ZScore":

                z = np.abs(stats.zscore(df[col]))
                outliers = df[z > 3]

            elif method == "IQR":

                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                outliers = df[
                    (df[col] < Q1 - 1.5*IQR) |
                    (df[col] > Q3 + 1.5*IQR)
                ]

            elif method == "Isolation Forest":

                model = IsolationForest()
                preds = model.fit_predict(df[numeric_cols])
                outliers = df[preds == -1]

            elif method == "LOF":

                model = LocalOutlierFactor()
                preds = model.fit_predict(df[numeric_cols])
                outliers = df[preds == -1]

            st.session_state.outliers = outliers

            st.session_state.results.append({
                "title": f"Outliers detected ({method})",
                "data": outliers
            })


    # ================================
    # OUTLIER HANDLING
    # ================================

    if module == "Outlier Handling":

        col = st.sidebar.selectbox("Column", numeric_cols)

        if st.sidebar.button("Apply Winsorization"):

            lower = df[col].quantile(0.05)
            upper = df[col].quantile(0.95)

            df[col] = df[col].clip(lower, upper)

            st.session_state.df = df

            st.session_state.results.append({
                "title": "Winsorization Applied",
                "data": df.head()
            })


    # ================================
    # SCALING
    # ================================

    if module == "Scaling":

        method = st.sidebar.selectbox(
            "Method",
            ["Standard", "MinMax", "Robust"]
        )

        if st.sidebar.button("Apply Scaling"):

            if method == "Standard":
                scaler = StandardScaler()

            elif method == "MinMax":
                scaler = MinMaxScaler()

            elif method == "Robust":
                scaler = RobustScaler()

            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

            st.session_state.df = df

            st.session_state.results.append({
                "title": f"Scaling Applied ({method})",
                "data": df.head()
            })


    # ================================
    # LINEAR REGRESSION
    # ================================

    if module == "Linear Regression":

        target = st.sidebar.selectbox("Target", numeric_cols)

        features = [col for col in numeric_cols if col != target]

        if st.sidebar.button("Run Regression"):

            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = LinearRegression()
            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            r2 = r2_score(y_test, pred)

            st.session_state.results.append({
                "title": "Linear Regression Results",
                "data": f"R2 Score: {r2}"
            })


    # ================================
    # THEIL SEN
    # ================================

    if module == "Theil-Sen Regression":

        target = st.sidebar.selectbox("Target", numeric_cols)

        features = [col for col in numeric_cols if col != target]

        if st.sidebar.button("Run Theil-Sen"):

            model = TheilSenRegressor()
            model.fit(df[features], df[target])

            st.session_state.results.append({
                "title": "Theil-Sen Results",
                "data": f"Slope: {model.coef_}"
            })


    # ================================
    # MANN KENDALL
    # ================================

    if module == "Mann-Kendall Test":

        col = st.sidebar.selectbox("Column", numeric_cols)

        if st.sidebar.button("Run Test"):

            result = mk.original_test(df[col])

            st.session_state.results.append({
                "title": "Mann Kendall Results",
                "data": result
            })


    # ================================
    # PCA
    # ================================

    if module == "PCA":

        if st.sidebar.button("Run PCA"):

            scaler = StandardScaler()

            scaled = scaler.fit_transform(df[numeric_cols])

            pca = PCA()

            pca.fit(scaled)

            st.session_state.results.append({
                "title": "PCA Variance Ratio",
                "data": pca.explained_variance_ratio_
            })


    # ================================
    # DISPLAY RESULTS HISTORY
    # ================================

    st.header("Analysis Results")

    for result in st.session_state.results:

        st.subheader(result["title"])
        st.write(result["data"])


    # ================================
    # DOWNLOAD
    # ================================

    st.sidebar.download_button(
        "Download Processed Data",
        df.to_csv(index=False),
        file_name="processed.csv"
    )

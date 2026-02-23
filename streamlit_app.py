# ============================================================
# PROFESSIONAL PREDICTIVE ANALYTICS DASHBOARD
# WITH AUTOMATIC INTERPRETATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import scipy.stats as stats

try:
    import pymannkendall as mk
except:
    mk = None


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(layout="wide")
st.title("Predictive Analytics Professional Dashboard")


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "results" not in st.session_state:
    st.session_state.results = []

if "outliers" not in st.session_state:
    st.session_state.outliers = None


# ============================================================
# FILE UPLOAD
# ============================================================

file = st.file_uploader("Upload CSV File")

if file:
    st.session_state.df = pd.read_csv(file)
    st.success("Dataset Loaded Successfully")

df = st.session_state.df


# ============================================================
# HELPER FUNCTION FOR INTERPRETATION
# ============================================================

def add_result(title, data):
    st.session_state.results.append((title, data))


# ============================================================
# MAIN MODULE
# ============================================================

if df is not None:

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    module = st.sidebar.selectbox(
        "Select Module",
        [
            "EDA",
            "Missing Values",
            "Outlier Detection",
            "Outlier Handling",
            "Scaling",
            "Linear Regression",
            "Theil-Sen Regression",
            "Mann-Kendall Test",
            "PCA"
        ]
    )


# ============================================================
# EDA
# ============================================================

    if module == "EDA":

        if st.sidebar.button("Run EDA"):

            add_result("Dataset Shape", df.shape)

            add_result("Missing Values", df.isnull().sum())

            add_result("Descriptive Statistics", df.describe())

            # Graph
            col = numeric_cols[0]

            fig, ax = plt.subplots()

            ax.hist(df[col])

            ax.set_title("Distribution")

            add_result("Histogram", fig)

            # Interpretation
            interpretation = """
EDA Interpretation:
• Provides understanding of dataset structure
• Helps identify missing values and anomalies
• Shows distribution of variables
"""

            add_result("EDA Interpretation", interpretation)


# ============================================================
# MISSING VALUES
# ============================================================

    if module == "Missing Values":

        if st.sidebar.button("Detect Missing"):

            missing = df.isnull().sum()

            add_result("Missing Values Count", missing)

            interpretation = f"""
Missing Values Interpretation:
• Total missing values: {missing.sum()}
• Missing values can reduce model accuracy
"""

            add_result("Missing Values Interpretation", interpretation)

        method = st.sidebar.selectbox(
            "Imputation Method",
            ["Mean", "Median", "KNN"]
        )

        if st.sidebar.button("Apply Imputation"):

            df_copy = df.copy()

            if method == "Mean":
                imputer = SimpleImputer(strategy="mean")

            elif method == "Median":
                imputer = SimpleImputer(strategy="median")

            else:
                imputer = KNNImputer()

            df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])

            st.session_state.df = df_copy

            add_result("Imputation Applied", df_copy.head())

            interpretation = f"""
Imputation Interpretation:
• Missing values filled using {method}
• Improves model reliability
"""

            add_result("Imputation Interpretation", interpretation)


# ============================================================
# OUTLIER DETECTION
# ============================================================

    if module == "Outlier Detection":

        col = st.sidebar.selectbox("Column", numeric_cols)

        if st.sidebar.button("Detect Outliers"):

            z = np.abs(stats.zscore(df[col]))

            outliers = df[z > 3]

            st.session_state.outliers = outliers

            count = len(outliers)

            add_result("Outlier Count", count)

            fig, ax = plt.subplots()

            ax.scatter(df.index, df[col])
            ax.scatter(outliers.index, outliers[col])

            add_result("Outlier Graph", fig)

            interpretation = f"""
Outlier Interpretation:
• Total outliers detected: {count}
• Outliers can distort regression models
"""

            add_result("Outlier Interpretation", interpretation)


# ============================================================
# LINEAR REGRESSION
# ============================================================

    if module == "Linear Regression":

        target = st.sidebar.selectbox("Target Variable", numeric_cols)

        features = st.sidebar.multiselect(
            "Independent Variables",
            [col for col in numeric_cols if col != target]
        )

        if st.sidebar.button("Run Regression"):

            if len(features) == 0:

                st.warning("Select independent variables")

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

                mse = mean_squared_error(y_test, pred)

                rmse = np.sqrt(mse)

                n = len(y_test)
                p = len(features)

                adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)

                coeff = pd.DataFrame({
                    "Variable": features,
                    "Coefficient": model.coef_
                })

                add_result("Regression Coefficients", coeff)

                metrics = pd.DataFrame({
                    "Metric": ["R²", "Adjusted R²", "RMSE"],
                    "Value": [r2, adj_r2, rmse]
                })

                add_result("Regression Metrics", metrics)

                fig, ax = plt.subplots()

                ax.scatter(y_test, pred)

                add_result("Regression Graph", fig)

                interpretation = f"""
Regression Interpretation:
• R² = {r2:.3f}
• Model explains {r2*100:.1f}% variance
• Higher R² indicates better prediction
"""

                add_result("Regression Interpretation", interpretation)


# ============================================================
# MANN KENDALL TEST
# ============================================================

    if module == "Mann-Kendall Test":

        col = st.sidebar.selectbox("Column", numeric_cols)

        if st.sidebar.button("Run Test"):

            if mk:

                result = mk.original_test(df[col])

                table = pd.DataFrame({
                    "Metric": ["Trend", "p-value", "Z"],
                    "Value": [result.trend, result.p, result.z]
                })

                add_result("Mann Kendall Results", table)

                if result.p < 0.05:

                    interpretation = f"""
Trend Interpretation:
• Trend is statistically significant (p={result.p:.4f})
• Strong evidence of trend
"""

                else:

                    interpretation = f"""
Trend Interpretation:
• No statistically significant trend (p={result.p:.4f})
"""

                add_result("Mann Kendall Interpretation", interpretation)

            fig, ax = plt.subplots()

            ax.plot(df[col])

            add_result("Trend Graph", fig)


# ============================================================
# PCA
# ============================================================

    if module == "PCA":

        if st.sidebar.button("Run PCA"):

            scaler = StandardScaler()

            scaled = scaler.fit_transform(df[numeric_cols])

            pca = PCA()

            pca.fit(scaled)

            variance = pca.explained_variance_ratio_

            add_result("Explained Variance", variance)

            fig, ax = plt.subplots()

            ax.plot(variance)

            add_result("Scree Plot", fig)

            interpretation = f"""
PCA Interpretation:
• First component explains {variance[0]*100:.2f}% variance
"""

            add_result("PCA Interpretation", interpretation)


# ============================================================
# DISPLAY RESULTS
# ============================================================

    st.header("Results")

    for title, result in st.session_state.results:

        st.subheader(title)

        if isinstance(result, plt.Figure):
            st.pyplot(result)

        else:
            st.write(result)


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

    def convert_results():

        dfs = []

        for title, result in st.session_state.results:

            if isinstance(result, pd.DataFrame):

                temp = result.copy()

                temp["Result"] = title

                dfs.append(temp)

        if dfs:

            return pd.concat(dfs)

        return pd.DataFrame()


    results_df = convert_results()

    st.sidebar.download_button(
        "Download Results",
        results_df.to_csv(index=False),
        "analysis_results.csv"
    )


    st.sidebar.download_button(
        "Download Processed Data",
        df.to_csv(index=False),
        "processed_data.csv"
    )

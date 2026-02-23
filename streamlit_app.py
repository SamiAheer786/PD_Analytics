# ==========================================
# PROFESSIONAL PREDICTIVE ANALYTICS APP
# ==========================================

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


# ==========================================
# STREAMLIT CONFIG
# ==========================================

st.set_page_config(layout="wide")
st.title("Predictive Analytics Professional Dashboard")


# ==========================================
# SESSION STATE
# ==========================================

if "df" not in st.session_state:
    st.session_state.df = None

if "results" not in st.session_state:
    st.session_state.results = []

if "outliers" not in st.session_state:
    st.session_state.outliers = None


# ==========================================
# FILE UPLOAD
# ==========================================

file = st.file_uploader("Upload CSV file")

if file:
    st.session_state.df = pd.read_csv(file)
    st.success("Data Loaded Successfully")

df = st.session_state.df


# ==========================================
# MAIN PROGRAM
# ==========================================

if df is not None:

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.sidebar.title("Modules")

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


# ==========================================
# EDA MODULE
# ==========================================

    if module == "EDA":

        if st.sidebar.button("Run EDA"):

            # Shape
            st.session_state.results.append(
                ("Dataset Shape", df.shape)
            )

            # Missing values
            missing = df.isnull().sum()

            st.session_state.results.append(
                ("Missing Values", missing)
            )

            # Descriptive stats
            st.session_state.results.append(
                ("Descriptive Statistics", df.describe())
            )

            # Histogram
            col = numeric_cols[0]

            fig, ax = plt.subplots()
            ax.hist(df[col], bins=30)
            ax.set_title(f"Distribution of {col}")

            st.session_state.results.append(
                ("Histogram", fig)
            )

            # Boxplot
            fig2, ax2 = plt.subplots()
            ax2.boxplot(df[col])
            ax2.set_title(f"Boxplot of {col}")

            st.session_state.results.append(
                ("Boxplot", fig2)
            )


# ==========================================
# MISSING VALUE MODULE
# ==========================================

    if module == "Missing Values":

        if st.sidebar.button("Detect Missing Values"):

            missing = df.isnull().sum()

            st.session_state.results.append(
                ("Missing Values Count", missing)
            )

            fig, ax = plt.subplots()

            ax.bar(missing.index, missing.values)

            plt.xticks(rotation=90)

            st.session_state.results.append(
                ("Missing Values Graph", fig)
            )

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

            st.session_state.results.append(
                ("Imputation Applied", df_copy.head())
            )


# ==========================================
# OUTLIER DETECTION
# ==========================================

    if module == "Outlier Detection":

        col = st.sidebar.selectbox("Column", numeric_cols)

        method = st.sidebar.selectbox(
            "Method",
            ["Z-score", "IQR", "Isolation Forest", "LOF"]
        )

        if st.sidebar.button("Detect Outliers"):

            if method == "Z-score":

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

            else:

                model = LocalOutlierFactor()

                preds = model.fit_predict(df[numeric_cols])

                outliers = df[preds == -1]

            count = len(outliers)

            st.session_state.outliers = outliers

            st.session_state.results.append(
                ("Outlier Count", count)
            )

            # Plot
            fig, ax = plt.subplots()

            ax.scatter(df.index, df[col])

            ax.scatter(outliers.index, outliers[col])

            ax.set_title("Outlier Visualization")

            st.session_state.results.append(
                ("Outlier Graph", fig)
            )


# ==========================================
# OUTLIER HANDLING
# ==========================================

    if module == "Outlier Handling":

        col = st.sidebar.selectbox("Column", numeric_cols)

        method = st.sidebar.selectbox(
            "Handling Method",
            ["Remove", "Winsorization", "Replace with Median"]
        )

        if st.sidebar.button("Apply Handling"):

            df_copy = df.copy()

            if method == "Remove":

                df_copy = df_copy.drop(st.session_state.outliers.index)

            elif method == "Winsorization":

                lower = df[col].quantile(0.05)
                upper = df[col].quantile(0.95)

                df_copy[col] = df_copy[col].clip(lower, upper)

            else:

                median = df[col].median()

                df_copy.loc[st.session_state.outliers.index, col] = median

            st.session_state.df = df_copy

            st.session_state.results.append(
                ("Outlier Handling Applied", df_copy.head())
            )


# ==========================================
# SCALING
# ==========================================

    if module == "Scaling":

        method = st.sidebar.selectbox(
            "Scaling Method",
            ["Standard", "MinMax", "Robust"]
        )

        if st.sidebar.button("Apply Scaling"):

            if method == "Standard":

                scaler = StandardScaler()

            elif method == "MinMax":

                scaler = MinMaxScaler()

            else:

                scaler = RobustScaler()

            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

            st.session_state.results.append(
                ("Scaling Applied", df.head())
            )


# ==========================================
# LINEAR REGRESSION
# ==========================================

    if module == "Linear Regression":

        target = st.sidebar.selectbox("Target", numeric_cols)

        features = [col for col in numeric_cols if col != target]

        if st.sidebar.button("Run Regression"):

            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2
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

            results = pd.DataFrame({
                "Feature": features,
                "Coefficient": model.coef_
            })

            st.session_state.results.append(
                ("Regression Coefficients", results)
            )

            st.session_state.results.append(
                ("Model Metrics", f"R2={r2}, AdjR2={adj_r2}, RMSE={rmse}")
            )

            fig, ax = plt.subplots()

            ax.scatter(y_test, pred)

            ax.set_title("Actual vs Predicted")

            st.session_state.results.append(
                ("Regression Graph", fig)
            )


# ==========================================
# THEIL SEN
# ==========================================

    if module == "Theil-Sen Regression":

        target = st.sidebar.selectbox("Target", numeric_cols)

        features = [col for col in numeric_cols if col != target]

        if st.sidebar.button("Run Theil-Sen"):

            model = TheilSenRegressor()

            model.fit(df[features], df[target])

            pred = model.predict(df[features])

            fig, ax = plt.subplots()

            ax.scatter(df[target], pred)

            ax.set_title("Theil-Sen Regression")

            st.session_state.results.append(
                ("Theil-Sen Graph", fig)
            )


# ==========================================
# MANN KENDALL
# ==========================================

    if module == "Mann-Kendall Test":

        col = st.sidebar.selectbox("Column", numeric_cols)

        if st.sidebar.button("Run Test"):

            if mk:

                result = mk.original_test(df[col])

                st.session_state.results.append(
                    ("Mann Kendall Result", result)
                )

            fig, ax = plt.subplots()

            ax.plot(df[col])

            ax.set_title("Trend Graph")

            st.session_state.results.append(
                ("Trend Graph", fig)
            )


# ==========================================
# PCA
# ==========================================

    if module == "PCA":

        if st.sidebar.button("Run PCA"):

            scaler = StandardScaler()

            scaled = scaler.fit_transform(df[numeric_cols])

            pca = PCA()

            pca.fit(scaled)

            fig, ax = plt.subplots()

            ax.plot(pca.explained_variance_ratio_)

            ax.set_title("Scree Plot")

            st.session_state.results.append(
                ("PCA Scree Plot", fig)
            )


# ==========================================
# SHOW RESULTS
# ==========================================

    st.header("Results")

    for title, result in st.session_state.results:

        st.subheader(title)

        if isinstance(result, plt.Figure):

            st.pyplot(result)

        else:

            st.write(result)


# ==========================================
# DOWNLOAD
# ==========================================

    st.sidebar.download_button(
        "Download Data",
        df.to_csv(index=False),
        file_name="processed.csv"
    )

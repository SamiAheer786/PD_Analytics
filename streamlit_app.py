import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

# Page setup
st.set_page_config(page_title="Predictive Analytics Control App", layout="wide")

st.title("Predictive Analytics Control Dashboard")

# Session state
if "df" not in st.session_state:
    st.session_state.df = None

# Upload file
uploaded_file = st.file_uploader("Upload CSV file")

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)

df = st.session_state.df

# Continue only if data exists
if df is not None:

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.sidebar.title("Control Panel")

    operation = st.sidebar.selectbox(
        "Choose Operation",
        [
            "EDA",
            "Imputation",
            "Outlier Detection",
            "Scaling"
        ]
    )

    # ========================
    # EDA
    # ========================
    if operation == "EDA":

        if st.sidebar.button("Run EDA"):

            st.subheader("Preview")
            st.write(df.head())

            st.subheader("Shape")
            st.write(df.shape)

            st.subheader("Missing values")
            st.write(df.isnull().sum())

            st.subheader("Summary")
            st.write(df.describe())

            column = st.selectbox("Select column for histogram", numeric_cols)

            if column:
                fig = plt.figure()
                plt.hist(df[column].dropna())
                plt.title(column)
                st.pyplot(fig)

    # ========================
    # IMPUTATION
    # ========================
    if operation == "Imputation":

        method = st.sidebar.selectbox(
            "Select Imputation Method",
            ["Mean", "Median", "Most Frequent", "KNN"]
        )

        if st.sidebar.button("Apply Imputation"):

            df_copy = df.copy()

            if method == "Mean":
                imputer = SimpleImputer(strategy="mean")
                df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])

            elif method == "Median":
                imputer = SimpleImputer(strategy="median")
                df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])

            elif method == "Most Frequent":
                imputer = SimpleImputer(strategy="most_frequent")
                df_copy[:] = imputer.fit_transform(df_copy)

            elif method == "KNN":
                imputer = KNNImputer()
                df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])

            st.session_state.df = df_copy

            st.success("Imputation Applied")
            st.write(df_copy.head())

    # ========================
    # OUTLIER DETECTION
    # ========================
    if operation == "Outlier Detection":

        column = st.sidebar.selectbox("Select Column", numeric_cols)

        method = st.sidebar.selectbox(
            "Select Method",
            [
                "Z-Score",
                "IQR",
                "Winsorization",
                "Isolation Forest",
                "LOF"
            ]
        )

        if st.sidebar.button("Run Outlier Method"):

            df_copy = df.copy()

            if method == "Z-Score":

                z = np.abs(stats.zscore(df_copy[column]))
                outliers = df_copy[z > 3]

                st.write("Outliers detected:", len(outliers))
                st.write(outliers)

            elif method == "IQR":

                Q1 = df_copy[column].quantile(0.25)
                Q3 = df_copy[column].quantile(0.75)
                IQR = Q3 - Q1

                outliers = df_copy[
                    (df_copy[column] < Q1 - 1.5 * IQR) |
                    (df_copy[column] > Q3 + 1.5 * IQR)
                ]

                st.write("Outliers detected:", len(outliers))
                st.write(outliers)

            elif method == "Winsorization":

                lower = df_copy[column].quantile(0.05)
                upper = df_copy[column].quantile(0.95)

                df_copy[column] = df_copy[column].clip(lower, upper)

                st.session_state.df = df_copy

                st.success("Winsorization Applied")
                st.write(df_copy.head())

            elif method == "Isolation Forest":

                model = IsolationForest(contamination=0.05)
                preds = model.fit_predict(df_copy[numeric_cols])

                outliers = df_copy[preds == -1]

                st.write("Outliers detected:", len(outliers))
                st.write(outliers)

            elif method == "LOF":

                model = LocalOutlierFactor()
                preds = model.fit_predict(df_copy[numeric_cols])

                outliers = df_copy[preds == -1]

                st.write("Outliers detected:", len(outliers))
                st.write(outliers)

    # ========================
    # SCALING
    # ========================
    if operation == "Scaling":

        method = st.sidebar.selectbox(
            "Select Scaling Method",
            [
                "StandardScaler",
                "MinMaxScaler",
                "RobustScaler"
            ]
        )

        if st.sidebar.button("Apply Scaling"):

            df_copy = df.copy()

            if method == "StandardScaler":
                scaler = StandardScaler()

            elif method == "MinMaxScaler":
                scaler = MinMaxScaler()

            elif method == "RobustScaler":
                scaler = RobustScaler()

            df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])

            st.session_state.df = df_copy

            st.success("Scaling Applied")
            st.write(df_copy.head())

    # ========================
    # DOWNLOAD
    # ========================
    st.sidebar.download_button(
        "Download Processed Data",
        df.to_csv(index=False),
        file_name="processed_data.csv"
    )

else:

    st.info("Upload CSV file to start.")

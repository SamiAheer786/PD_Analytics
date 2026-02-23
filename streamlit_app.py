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

import scipy.stats as stats

# Safe Mann-Kendall
try:
    import pymannkendall as mk
except:
    mk = None

# Safe PDF import
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False


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
# HELPER FUNCTION
# ============================================================

def add_result(title, data):
    st.session_state.results.append((title, data))


# ============================================================
# MAIN LOGIC
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

    # ======================= EDA =======================

    if module == "EDA":

        if st.sidebar.button("Run EDA"):
            add_result("Dataset Shape", df.shape)
            add_result("Missing Values", df.isnull().sum())
            add_result("Descriptive Statistics", df.describe())

            if numeric_cols:
                col = numeric_cols[0]
                fig, ax = plt.subplots()
                ax.hist(df[col])
                ax.set_title("Distribution")
                add_result("Histogram", fig)

            add_result("EDA Interpretation",
                       "EDA helps understand structure, missing data and variable distribution.")

    # ======================= MISSING VALUES =======================

    if module == "Missing Values":

        if st.sidebar.button("Detect Missing"):
            add_result("Missing Values Count", df.isnull().sum())

        method = st.sidebar.selectbox("Imputation Method", ["Mean", "Median", "KNN"])

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

    # ======================= OUTLIER DETECTION =======================

    if module == "Outlier Detection":

        col = st.sidebar.selectbox("Column", numeric_cols)

        if st.sidebar.button("Detect Outliers"):

            z = np.abs(stats.zscore(df[col]))
            outliers = df[z > 3]
            st.session_state.outliers = outliers

            add_result("Outlier Count", len(outliers))

            fig, ax = plt.subplots()
            ax.scatter(df.index, df[col])
            ax.scatter(outliers.index, outliers[col])
            ax.set_title("Outlier Detection")
            add_result("Outlier Graph", fig)

    # ======================= OUTLIER HANDLING =======================

    if module == "Outlier Handling":

        method = st.sidebar.selectbox("Handling Method", ["Remove", "Cap"])

        if st.sidebar.button("Apply Outlier Handling"):

            if st.session_state.outliers is None:
                st.warning("Run Outlier Detection first")
            else:
                df_copy = df.copy()

                if method == "Remove":
                    df_copy = df_copy.drop(st.session_state.outliers.index)
                else:
                    for col in numeric_cols:
                        lower = df_copy[col].quantile(0.01)
                        upper = df_copy[col].quantile(0.99)
                        df_copy[col] = np.clip(df_copy[col], lower, upper)

                st.session_state.df = df_copy
                add_result("Outliers Handled", df_copy.head())

    # ======================= SCALING =======================

    if module == "Scaling":

        scaler_type = st.sidebar.selectbox("Select Scaler", ["Standard", "MinMax", "Robust"])

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
            add_result("Scaling Applied", df_copy.head())

    # ======================= LINEAR REGRESSION =======================

    if module == "Linear Regression":

        target = st.sidebar.selectbox("Target Variable", numeric_cols)
        features = st.sidebar.multiselect(
            "Independent Variables",
            [col for col in numeric_cols if col != target]
        )

        if st.sidebar.button("Run Regression"):

            if not features:
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
                rmse = np.sqrt(mean_squared_error(y_test, pred))

                add_result("R2 Score", r2)
                add_result("RMSE", rmse)
                add_result("Intercept", model.intercept_)
                add_result("Coefficients", dict(zip(features, model.coef_)))

                fig, ax = plt.subplots()
                ax.scatter(y_test, pred)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                add_result("Regression Plot", fig)

                add_result("Interpretation",
                           f"The model explains {round(r2*100,2)}% variance. "
                           f"RMSE is {round(rmse,2)}.")

    # ======================= THEIL-SEN =======================

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

            coef = model.coef_[0]
            intercept = model.intercept_

            add_result("Slope", coef)
            add_result("Intercept", intercept)

            fig, ax = plt.subplots()
            ax.scatter(X, y)
            ax.plot(X, model.predict(X))
            ax.set_title("Theil-Sen Regression")
            add_result("Theil-Sen Plot", fig)

            add_result("Interpretation",
                       f"Each unit increase in {feature} changes {target} by {round(coef,3)}.")

    # ======================= MANN-KENDALL =======================

    if module == "Mann-Kendall Test":

        col = st.sidebar.selectbox("Column", numeric_cols)

        if st.sidebar.button("Run Test"):

            if mk:
                result = mk.original_test(df[col])

                add_result("Trend", result.trend)
                add_result("p-value", result.p)
                add_result("Tau", result.Tau)
                add_result("Slope", result.slope)

                fig, ax = plt.subplots()
                ax.plot(df[col])
                ax.set_title("Trend Visualization")
                add_result("Trend Plot", fig)

                if result.p < 0.05:
                    add_result("Interpretation", "Statistically significant trend detected.")
                else:
                    add_result("Interpretation", "No statistically significant trend detected.")
            else:
                st.warning("pymannkendall not installed")

    # ======================= PCA =======================

    if module == "PCA":

        if st.sidebar.button("Run PCA"):

            scaler = StandardScaler()
            scaled = scaler.fit_transform(df[numeric_cols])

            pca = PCA()
            pca.fit(scaled)

            add_result("Explained Variance Ratio", pca.explained_variance_ratio_)

    # ======================= DISPLAY RESULTS =======================

    st.header("Results")

    for title, result in st.session_state.results:
        st.subheader(title)
        if isinstance(result, plt.Figure):
            st.pyplot(result)
        else:
            st.write(result)

    # ======================= PDF DOWNLOAD =======================

    if st.sidebar.button("Download Results as PDF"):

        if not PDF_AVAILABLE:
            st.sidebar.error("Install reportlab in requirements.txt")
        else:
            file_path = "analysis_report.pdf"
            doc = SimpleDocTemplate(file_path)
            elements = []
            styles = getSampleStyleSheet()

            elements.append(Paragraph("Predictive Analytics Report", styles["Heading1"]))
            elements.append(Spacer(1, 0.3 * inch))

            for title, result in st.session_state.results:
                elements.append(Paragraph(f"<b>{title}</b>", styles["Normal"]))
                elements.append(Spacer(1, 0.1 * inch))
                elements.append(Paragraph(str(result), styles["Normal"]))
                elements.append(Spacer(1, 0.2 * inch))

            doc.build(elements)

            with open(file_path, "rb") as f:
                st.sidebar.download_button(
                    "Click to Download PDF",
                    f,
                    file_name="analysis_report.pdf"
                )

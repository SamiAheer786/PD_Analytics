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

# Safe PDF Import (prevents crash)
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
# OUTLIER HANDLING (NEW - FIXED)
# ============================================================

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
                st.success("Outliers handled successfully")


# ============================================================
# SCALING (NEW - FIXED)
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
            add_result("Scaling Applied", df_copy.head())
            st.success("Scaling applied successfully")


# ============================================================
# THEIL-SEN REGRESSION (NEW - FIXED)
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

            coeff = model.coef_[0]
            intercept = model.intercept_

            table = pd.DataFrame({
                "Metric": ["Coefficient", "Intercept"],
                "Value": [coeff, intercept]
            })

            add_result("Theil-Sen Regression Results", table)

            interpretation = f"""
Theil-Sen Interpretation:
• Robust regression method
• Coefficient = {coeff:.4f}
• Less sensitive to outliers
"""
            add_result("Theil-Sen Interpretation", interpretation)


# ============================================================
# PDF DOWNLOAD (REPLACES CSV)
# ============================================================

    if st.sidebar.button("Download Results as PDF"):

        if not PDF_AVAILABLE:
            st.sidebar.error("Add reportlab to requirements.txt")
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

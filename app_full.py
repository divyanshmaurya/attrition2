
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import joblib

from pathlib import Path

st.set_page_config(page_title="Employee Attrition — Full Dashboard", page_icon="💼", layout="wide")

DATA_PATH = Path("Data.csv")
MODEL_PATH = Path("best_model.pkl")
SCORES_PATH = Path("model_scores.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_scores():
    try:
        return pd.read_csv(SCORES_PATH, index_col=0)
    except Exception:
        return pd.DataFrame()

def to_binary(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.strip()
    return (s.isin(["yes","true","1","leave","left"])).astype(int)

def get_proba(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == 2:
            return float(p[:,1][0])
        return float(np.max(p[0]))
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        return 1.0 / (1.0 + np.exp(-score))
    pred = model.predict(X)[0]
    return float(pred) if pred in [0,1] else 0.0

data = load_data()
model = load_model()
scores = load_scores()

target_candidates = ["Attrition","attrition","TARGET","target","label","Label"]
target_col = next((c for c in target_candidates if c in data.columns), None)
if target_col is None:
    st.error("Target column not found. Please include one of: " + ", ".join(target_candidates))
    st.stop()

X_all = data.drop(columns=[target_col])
y_raw = data[target_col]
y_all = y_raw if y_raw.dtype != object else to_binary(y_raw)

num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_all.columns if c not in num_cols]

st.title("💼 Employee Attrition — Analytics & Simulation Dashboard")

tab_overview, tab_models, tab_eda, tab_predict = st.tabs(["🏠 Overview", "📊 Model Comparison", "📈 Explore Data", "🔮 Predict & Sensitivity"])

with tab_overview:
    st.subheader("Dataset Snapshot")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.metric("Total Records", f"{len(data):,}")
    with c2:
        st.metric("Attrition Rate", f"{float(pd.Series(y_all).mean())*100:.1f}%")
    with c3:
        st.metric("Avg Monthly Income", f"${data['MonthlyIncome'].mean():,.0f}" if "MonthlyIncome" in data.columns else "—")
    with c4:
        st.metric("Avg Years at Company", f"{data['YearsAtCompany'].mean():.1f}" if "YearsAtCompany" in data.columns else "—")

    st.markdown("### Attrition Distribution")
    dist_df = y_raw.value_counts(dropna=False).rename_axis("Attrition").reset_index(name="Count")
    chart = alt.Chart(dist_df).mark_bar().encode(
        x=alt.X("Attrition:N", title="Attrition"),
        y=alt.Y("Count:Q", title="Count"),
        tooltip=["Attrition","Count"]
    )
    st.altair_chart(chart, use_container_width=True)

with tab_models:
    st.subheader("Model Performance (from training)")
    if scores.empty:
        st.info("model_scores.csv not found.")
    else:
        st.dataframe(scores, use_container_width=True)
        metric_col = None
        for cand in ["Accuracy","F1","Precision","Recall"]:
            if cand in scores.columns:
                metric_col = cand
                break
        if metric_col:
            chart_df = scores.reset_index().rename(columns={"index":"Model"})
            bar = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Model:N", sort="-y"),
                y=alt.Y(f"{metric_col}:Q", title=metric_col),
                tooltip=["Model", metric_col]
            )
            st.altair_chart(bar, use_container_width=True)

with tab_eda:
    st.subheader("Explore Data")
    st.dataframe(data.head(50), use_container_width=True)

    if len(num_cols) > 0:
        num_feat = st.selectbox("Numeric feature", options=num_cols)
        hist = alt.Chart(data).mark_bar().encode(
            x=alt.X(f"{num_feat}:Q", bin=alt.Bin(maxbins=30), title=num_feat),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[num_feat, "count()"]
        )
        st.altair_chart(hist, use_container_width=True)
    else:
        st.info("No numeric columns found.")

    if target_col in data.columns and len(cat_cols) > 0:
        cat_feat = st.selectbox("Categorical feature", options=cat_cols)
        grp = data.groupby(cat_feat)[target_col].apply(lambda s: (s.astype(str).str.lower().isin(["yes","true","1","leave","left"])).mean()).reset_index(name="AttritionRate")
        bar = alt.Chart(grp).mark_bar().encode(
            x=alt.X(f"{cat_feat}:N", sort="-y", title=cat_feat),
            y=alt.Y("AttritionRate:Q", title="Attrition Rate"),
            tooltip=[cat_feat, alt.Tooltip("AttritionRate:Q", format=".2f")]
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("No categorical columns or target not found.")

with tab_predict:
    st.subheader("Predict Attrition & Sensitivity")

    numeric_defaults = {
        "Age": (18, 60, 30),
        "MonthlyIncome": (1000, 25000, 5000),
        "DistanceFromHome": (0, 50, 10),
        "TotalWorkingYears": (0, 40, 8),
        "YearsAtCompany": (0, 40, 5),
        "YearsInCurrentRole": (0, 20, 3),
        "YearsSinceLastPromotion": (0, 20, 1),
        "YearsWithCurrManager": (0, 20, 2),
        "NumCompaniesWorked": (0, 15, 2),
        "PercentSalaryHike": (0, 100, 15),
        "TrainingTimesLastYear": (0, 20, 3),
        "JobLevel": (1, 5, 2),
        "Education": (1, 5, 3)
    }
    cat_defaults = {
        "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        "Department": ["Sales", "Research & Development", "Human Resources"],
        "EducationField": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        "Gender": ["Male", "Female"],
        "JobRole": ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
                    "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"],
        "MaritalStatus": ["Single", "Married", "Divorced"],
        "OverTime": ["No", "Yes"]
    }

    user_inputs = {}
    c1, c2 = st.columns(2)
    for i, col in enumerate(num_cols):
        rng = numeric_defaults.get(col, (0, 100, 10))
        with (c1 if i % 2 == 0 else c2):
            user_inputs[col] = st.slider(f"{col}", min_value=float(rng[0]), max_value=float(rng[1]), value=float(rng[2]))

    c3, c4 = st.columns(2)
    for i, col in enumerate(cat_cols):
        opts = cat_defaults.get(col, ["Unknown", "Value1", "Value2"])
        with (c3 if i % 2 == 0 else c4):
            user_inputs[col] = st.selectbox(f"{col}", options=opts)

    input_df = pd.DataFrame([user_inputs])

    if st.button("🔮 Predict"):
        try:
            proba = get_proba(model, input_df)
            label = "Will Leave" if proba >= 0.5 else "Will Stay"
            st.metric("Attrition Probability", f"{proba*100:.1f}%")
            st.markdown(f"**Predicted Class:** :{'red_circle' if label=='Will Leave' else 'green_circle'}: **{label}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.markdown("### Sensitivity Analysis")
    left, right = st.columns([1,3])
    with left:
        sweep_feature = st.selectbox("Feature to vary", options=num_cols + cat_cols)
        points = st.slider("Resolution", min_value=5, max_value=50, value=25)
    with right:
        try:
            if sweep_feature in num_cols:
                rng = numeric_defaults.get(sweep_feature, (0, 100, 10))
                xs = np.linspace(rng[0], rng[1], points)
                probs = []
                for val in xs:
                    row = input_df.copy()
                    row[sweep_feature] = float(val)
                    probs.append(get_proba(model, row))
                line_df = pd.DataFrame({sweep_feature: xs, "Probability": probs})
                line = alt.Chart(line_df).mark_line().encode(
                    x=alt.X(f"{sweep_feature}:Q", title=sweep_feature),
                    y=alt.Y("Probability:Q", title="Attrition Probability"),
                    tooltip=[sweep_feature, alt.Tooltip("Probability:Q", format=".2f")]
                )
                st.altair_chart(line, use_container_width=True)
            else:
                opts = cat_defaults.get(sweep_feature, ["Unknown", "Value1", "Value2"])
                probs = []
                for val in opts:
                    row = input_df.copy()
                    row[sweep_feature] = val
                    probs.append(get_proba(model, row))
                line_df = pd.DataFrame({sweep_feature: opts, "Probability": probs})
                line = alt.Chart(line_df).mark_line(point=True).encode(
                    x=alt.X(f"{sweep_feature}:N", title=sweep_feature),
                    y=alt.Y("Probability:Q", title="Attrition Probability"),
                    tooltip=[sweep_feature, alt.Tooltip("Probability:Q", format=".2f")]
                )
                st.altair_chart(line, use_container_width=True)
        except Exception as e:
            st.error(f"Sensitivity analysis failed: {e}")

"""
app.py
======
Interactive Road Safety Dashboard - Ottawa transportation collision analysis
with BI dashboard + predictive analytics.
CST2213 - Business Intelligence Programming 2: Advanced Concepts - Final Project.
Student: Mohamed Samatar | Dataset: City of Ottawa Open Data, 2020 Tabular
Transportation Collision Data (10,047 records, all severities).

HOW TO RUN
----------
1. Put this file (app.py) and 2020_Tabular_Transportation_Collision_Data.csv
   in the SAME folder. Nothing else is required.
2. Open a command prompt / terminal in that folder.
3. Run:  pip install streamlit pandas numpy scikit-learn plotly joblib
4. Run:  streamlit run app.py

Single, self-contained file on purpose - no src/ package, no separate
train.py - so there is nothing to import incorrectly and nothing to break
by running it from the wrong folder.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Interactive Road Safety Dashboard", layout="wide", page_icon="🚦")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_NAME = "2020_Tabular_Transportation_Collision_Data.csv"

NUMERIC_FEATURES = ["Hour", "Month"]
CATEGORICAL_FEATURES = ["Env", "Light_Cond", "Road", "Traffic", "Impact", "Loc_Type"]
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

REQUIRED_RAW_COLUMNS = [
    "Accident_Date", "Accident_Time", "Accident_Location",
    "Classification_of_Accident", "Initial_Impact_Type", "Environment_Condition",
    "Light", "Road_Surface_Condition", "Traffic_Control", "Traffic_Control_Condition",
    "No__of_Vehicles", "No__of_Bicycles", "No__of_Motorcycles", "No__of_Pedestrians",
    "Max_Injury", "No__of_Injuries", "No__of_Minimal", "No__of_Minor", "No__of_Major",
    "No__of_Fatal", "Latitude", "Longitude", "ObjectId",
]


# ============================================================== #
# 1. DATA LOADING + CLEANING + FEATURE ENGINEERING (all inline)
# ============================================================== #
def simplify_code(val) -> str:
    """Strip a leading numeric code, e.g. '01 - Clear' -> 'Clear'."""
    if pd.isna(val):
        return "Unknown"
    return val.split(" - ")[-1] if " - " in str(val) else str(val)


def validate_schema(df: pd.DataFrame) -> None:
    """Raise a clear, readable error if the uploaded/loaded CSV isn't the
    original raw City of Ottawa export this app was built for."""
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "This file doesn't match the expected raw collision data format.\n\n"
            f"Missing column(s): {', '.join(missing)}\n\n"
            "This app expects the ORIGINAL raw file exactly as downloaded from "
            "the City of Ottawa Open Data portal "
            "(2020_Tabular_Transportation_Collision_Data.csv) - not a cleaned, "
            "renamed, or filtered version."
        )


@st.cache_data(show_spinner="Loading and cleaning collision data...")
def load_and_clean(csv_path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(csv_path_or_buffer, encoding="utf-8-sig")
    validate_schema(df)

    # --- Missing value treatment (structural, not data-entry errors) ---
    df["Traffic_Control_Condition"] = df["Traffic_Control_Condition"].fillna("Not Applicable")
    injury_cols = ["No__of_Minimal", "No__of_Minor", "No__of_Major", "No__of_Fatal", "Max_Injury"]
    df[injury_cols] = df[injury_cols].fillna(0)

    # --- Duplicate check ---
    df = df.drop_duplicates(subset=["ObjectId"])

    # --- Coordinate sanity check (Ottawa bounding box) ---
    df["Coord_Valid"] = df["Latitude"].between(45.0, 45.7) & df["Longitude"].between(-76.2, -75.2)

    # --- Date / time parsing ---
    df["Accident_Date"] = pd.to_datetime(df["Accident_Date"])
    df["Accident_Time"] = pd.to_datetime(df["Accident_Time"], format="%I:%M %p", errors="coerce")
    df["Month_Name"] = df["Accident_Date"].dt.month_name()
    df["Month"] = df["Accident_Date"].dt.month
    df["Day_of_Week"] = df["Accident_Date"].dt.day_name()
    df["Hour"] = df["Accident_Time"].dt.hour

    # --- Decode coded categorical fields ("01 - Clear" -> "Clear") ---
    df["Env"] = df["Environment_Condition"].apply(simplify_code)
    df["Light_Cond"] = df["Light"].apply(simplify_code)
    df["Road"] = df["Road_Surface_Condition"].apply(simplify_code)
    df["Impact"] = df["Initial_Impact_Type"].apply(simplify_code)
    df["Traffic"] = df["Traffic_Control"].apply(simplify_code)
    df["Loc_Type"] = df["Accident_Location"].apply(simplify_code)

    # --- Severity label + binary ML target ---
    sev_map = {
        "01 - Fatal injury": "Fatal",
        "02 - Non-fatal injury": "Non-Fatal Injury",
        "03 - P.D. only": "Property Damage Only",
    }
    df["Severity"] = df["Classification_of_Accident"].map(sev_map)
    df["Injury_Event"] = df["Severity"].isin(["Fatal", "Non-Fatal Injury"]).astype(int)

    return df


# ============================================================== #
# 2. PREDICTIVE MODEL (trained in-memory, cached across reruns)
# ============================================================== #
def prepare_xy(df: pd.DataFrame):
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["Injury_Event"]
    model_df = df[cols].dropna()
    encoded = pd.get_dummies(model_df, columns=CATEGORICAL_FEATURES)
    X = encoded.drop(columns=["Injury_Event"])
    y = encoded["Injury_Event"]
    return X, y


@st.cache_resource(show_spinner="Training predictive models (this runs once)...")
def train_and_select_model(df: pd.DataFrame):
    """Trains Logistic Regression, Random Forest, and Gradient Boosting,
    compares them on a held-out test set, and keeps the one with the best
    recall on the Injury class (missing a risky condition is costlier than
    a false alarm for a road-safety tool)."""
    X, y = prepare_xy(df)
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    rows = []
    best_model, best_name, best_recall = None, None, -1.0
    best_cm = None

    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        y_prob = estimator.predict_proba(X_test)[:, 1]

        report = classification_report(
            y_test, y_pred, target_names=["No Injury (0)", "Injury (1)"], output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        acc = (y_pred == y_test).mean()

        rows.append({
            "Model": name,
            "Accuracy": round(acc, 3),
            "ROC-AUC": round(auc, 3),
            "Injury Precision": round(report["Injury (1)"]["precision"], 3),
            "Injury Recall": round(report["Injury (1)"]["recall"], 3),
            "Injury F1": round(report["Injury (1)"]["f1-score"], 3),
        })

        if report["Injury (1)"]["recall"] > best_recall:
            best_recall = report["Injury (1)"]["recall"]
            best_model, best_name = estimator, name
            best_cm = cm

    comparison_df = pd.DataFrame(rows).sort_values("Injury Recall", ascending=False)

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    elif hasattr(best_model, "coef_"):
        importances = pd.Series(np.abs(best_model.coef_[0]), index=feature_columns).sort_values(ascending=False)
    else:
        importances = None

    return {
        "model": best_model,
        "name": best_name,
        "feature_columns": feature_columns,
        "comparison": comparison_df,
        "confusion_matrix": best_cm,
        "importances": importances,
    }


def predict_risk(model_bundle, scenario: dict) -> float:
    row = {col: 0 for col in model_bundle["feature_columns"]}
    for feat in NUMERIC_FEATURES:
        if feat in row:
            row[feat] = scenario.get(feat, 0)
    for cat_col in CATEGORICAL_FEATURES:
        dummy_col = f"{cat_col}_{scenario.get(cat_col)}"
        if dummy_col in row:
            row[dummy_col] = 1
    X_scenario = pd.DataFrame([row])[model_bundle["feature_columns"]]
    return float(model_bundle["model"].predict_proba(X_scenario)[0, 1])


def risk_tier(probability: float) -> str:
    if probability < 0.15:
        return "Low"
    elif probability < 0.30:
        return "Medium"
    return "High"


# ============================================================== #
# 3. GEOGRAPHIC HOTSPOT CLUSTERING
# ============================================================== #
@st.cache_resource(show_spinner="Clustering geographic hotspots...")
def build_hotspots(df: pd.DataFrame, n_clusters: int = 25):
    geo_df = df[df["Coord_Valid"]].dropna(subset=["Latitude", "Longitude"]).copy()
    coords = geo_df[["Latitude", "Longitude"]].values

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    geo_df["Cluster"] = km.fit_predict(coords)

    summary = (
        geo_df.groupby("Cluster")
        .agg(
            lat=("Latitude", "mean"),
            lon=("Longitude", "mean"),
            collision_count=("Injury_Event", "count"),
            injury_count=("Injury_Event", "sum"),
            fatal_count=("Severity", lambda s: (s == "Fatal").sum()),
        )
        .reset_index()
        .rename(columns={"Cluster": "cluster_id"})
    )
    summary["injury_rate_pct"] = (100 * summary["injury_count"] / summary["collision_count"]).round(1)
    return summary.sort_values("collision_count", ascending=False).reset_index(drop=True)


# ============================================================== #
# 4. SIDEBAR - DATA INPUT
# ============================================================== #
st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

try:
    if uploaded is not None:
        df = load_and_clean(uploaded)
    else:
        default_path = os.path.join(SCRIPT_DIR, DEFAULT_CSV_NAME)
        if not os.path.exists(default_path):
            st.error(
                f"Could not find '{DEFAULT_CSV_NAME}' in the same folder as this script.\n\n"
                f"Looked here: {default_path}\n\n"
                "Either place the CSV next to app.py, or upload a file using the sidebar."
            )
            st.stop()
        df = load_and_clean(default_path)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not load this file: {e}")
    st.stop()

model_bundle = train_and_select_model(df)
hotspot_summary = build_hotspots(df, n_clusters=25)


# ============================================================== #
# 5. SIDEBAR - FILTERS (single-select "All" dropdowns)
# ============================================================== #
st.sidebar.header("Filters")

weather_options = ["All"] + sorted(df["Env"].dropna().unique().tolist())
selected_weather = st.sidebar.selectbox("Weather", weather_options)

light_options = ["All"] + sorted(df["Light_Cond"].dropna().unique().tolist())
selected_light = st.sidebar.selectbox("Light Condition", light_options)

road_options = ["All"] + sorted(df["Road"].dropna().unique().tolist())
selected_road = st.sidebar.selectbox("Road Condition", road_options)

filtered_df = df.copy()
if selected_weather != "All":
    filtered_df = filtered_df[filtered_df["Env"] == selected_weather]
if selected_light != "All":
    filtered_df = filtered_df[filtered_df["Light_Cond"] == selected_light]
if selected_road != "All":
    filtered_df = filtered_df[filtered_df["Road"] == selected_road]


# ============================================================== #
# 6. HEADER + KPIs
# ============================================================== #
st.title("Interactive Road Safety Dashboard")
st.caption("Ottawa transportation collision analysis with BI dashboard + predictive analytics")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Records", f"{len(filtered_df):,}")
k2.metric("Columns", filtered_df.shape[1])
k3.metric("Missing Values", int(filtered_df.isna().sum().sum()))
k4.metric("Injury Rate", f"{100 * filtered_df['Injury_Event'].mean():.1f}%" if len(filtered_df) else "0.0%")

tab_overview, tab_trend, tab_condition, tab_model, tab_predict = st.tabs(
    ["Overview", "Trend Analysis", "Condition Analysis", "Model Performance", "Risk Prediction"]
)


# ============================================================== #
# TAB 1 - Overview
# ============================================================== #
with tab_overview:
    st.subheader("Dataset Overview")
    display_cols = [
        "Accident_Date", "Hour", "Loc_Type", "Env", "Light_Cond", "Road",
        "Severity", "Impact", "Traffic",
    ]
    st.dataframe(filtered_df[display_cols].head(20), use_container_width=True)

    st.subheader("Missing Values Summary")
    missing_df = filtered_df.isna().sum().reset_index()
    missing_df.columns = ["column", "missing_count"]
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values("missing_count", ascending=False)
    if missing_df.empty:
        st.success("No missing values remain after cleaning.")
    else:
        st.dataframe(missing_df, use_container_width=True)

    st.subheader("Severity Breakdown")
    sev_counts = filtered_df["Severity"].value_counts().reset_index()
    sev_counts.columns = ["Severity", "Count"]
    fig_sev = px.bar(sev_counts, x="Severity", y="Count", title="Collisions by Severity")
    st.plotly_chart(fig_sev, use_container_width=True)


# ============================================================== #
# TAB 2 - Trend Analysis
# ============================================================== #
with tab_trend:
    st.subheader("Trend Analysis")

    if filtered_df.empty:
        st.info("No records match the current filters.")
    else:
        month_counts = (
            filtered_df["Month_Name"].value_counts().reindex(MONTH_ORDER).fillna(0).reset_index()
        )
        month_counts.columns = ["month", "count"]
        fig_month = px.bar(month_counts, x="month", y="count", title="Collisions by Month")
        st.plotly_chart(fig_month, use_container_width=True)

        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = (
            filtered_df["Day_of_Week"].value_counts().reindex(dow_order).fillna(0).reset_index()
        )
        dow_counts.columns = ["day_of_week", "count"]
        fig_dow = px.bar(dow_counts, x="day_of_week", y="count", title="Collisions by Day of Week")
        st.plotly_chart(fig_dow, use_container_width=True)

        hour_counts = filtered_df.groupby("Hour").size().reset_index(name="count")
        fig_hour = px.line(hour_counts, x="Hour", y="count", markers=True, title="Collisions by Hour")
        fig_hour.add_vrect(x0=14, x1=18, fillcolor="orange", opacity=0.15, line_width=0)
        st.plotly_chart(fig_hour, use_container_width=True)


# ============================================================== #
# TAB 3 - Condition Analysis
# ============================================================== #
with tab_condition:
    st.subheader("Condition Analysis")

    if filtered_df.empty:
        st.info("No records match the current filters.")
    else:
        weather_severity = pd.crosstab(filtered_df["Env"], filtered_df["Severity"]).reset_index()
        fig_weather = px.bar(weather_severity, x="Env", y=weather_severity.columns[1:],
                              barmode="group", title="Weather vs Severity")
        st.plotly_chart(fig_weather, use_container_width=True)

        light_severity = pd.crosstab(filtered_df["Light_Cond"], filtered_df["Severity"]).reset_index()
        fig_light = px.bar(light_severity, x="Light_Cond", y=light_severity.columns[1:],
                            barmode="group", title="Light Condition vs Severity")
        st.plotly_chart(fig_light, use_container_width=True)

        road_severity = pd.crosstab(filtered_df["Road"], filtered_df["Severity"]).reset_index()
        fig_road = px.bar(road_severity, x="Road", y=road_severity.columns[1:],
                           barmode="group", title="Road Condition vs Severity")
        st.plotly_chart(fig_road, use_container_width=True)

        top_locations = filtered_df["Loc_Type"].astype(str).value_counts().head(10).reset_index()
        top_locations.columns = ["Loc_Type", "count"]
        fig_loc = px.bar(top_locations, x="Loc_Type", y="count", title="Top Collision Location Types")
        st.plotly_chart(fig_loc, use_container_width=True)

        st.markdown("#### Geographic Hotspot Map")
        st.caption("K-Means clusters. Bubble size = collision count, color = injury rate.")
        fig_map = px.scatter_mapbox(
            hotspot_summary, lat="lat", lon="lon", size="collision_count", color="injury_rate_pct",
            color_continuous_scale="YlOrRd", size_max=40, zoom=9.3,
            center={"lat": 45.38, "lon": -75.70}, mapbox_style="carto-positron",
            hover_data={"collision_count": True, "injury_count": True, "fatal_count": True,
                        "injury_rate_pct": True, "lat": False, "lon": False},
            labels={"injury_rate_pct": "Injury Rate (%)", "collision_count": "Collisions"},
        )
        fig_map.update_layout(height=500, margin=dict(t=10, b=10, l=0, r=0))
        st.plotly_chart(fig_map, use_container_width=True)


# ============================================================== #
# TAB 4 - Model Performance
# ============================================================== #
with tab_model:
    st.subheader("Model Performance")

    st.success(f"Model trained successfully. Best model: {model_bundle['name']}")

    st.markdown("#### Model Comparison (held-out test set)")
    st.dataframe(model_bundle["comparison"].reset_index(drop=True), use_container_width=True)
    st.caption(
        "Best model selected by **Injury-class recall** - missing an at-risk "
        "condition is costlier than a false alarm for a road-safety tool."
    )

    st.markdown("#### Confusion Matrix")
    cm = model_bundle["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual: No Injury", "Actual: Injury"],
                          columns=["Predicted: No Injury", "Predicted: Injury"])
    st.dataframe(cm_df, use_container_width=True)

    if model_bundle["importances"] is not None:
        st.markdown("#### Feature Importance")
        top_features = model_bundle["importances"].head(12)
        clean_labels = [f.split("_", 1)[-1] if "_" in f else f for f in top_features.index]
        fig_imp = px.bar(x=top_features.values, y=clean_labels, orientation="h",
                          labels={"x": "Importance", "y": "Feature"},
                          title=f"Top 12 Feature Importances ({model_bundle['name']})")
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)


# ============================================================== #
# TAB 5 - Risk Prediction
# ============================================================== #
with tab_predict:
    st.subheader("Risk Prediction Tool")
    st.write("Enter conditions below to predict the probability of an injury or fatal collision.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sim_month = st.selectbox("Month", options=list(range(1, 13)), format_func=lambda m: MONTH_ORDER[m - 1])
        sim_hour = st.slider("Hour of day", 0, 23, 16)
    with c2:
        sim_env = st.selectbox("Weather", options=sorted(df["Env"].dropna().unique()))
        sim_light = st.selectbox("Light condition", options=sorted(df["Light_Cond"].dropna().unique()))
    with c3:
        sim_road = st.selectbox("Road surface", options=sorted(df["Road"].dropna().unique()))
        sim_traffic = st.selectbox("Traffic control", options=sorted(df["Traffic"].dropna().unique()))
    with c4:
        sim_impact = st.selectbox("Impact type", options=sorted(df["Impact"].dropna().unique()))
        sim_loc = st.selectbox("Location type", options=sorted(df["Loc_Type"].dropna().unique()))

    if st.button("Predict Severity", type="primary"):
        scenario = {
            "Month": sim_month, "Hour": sim_hour, "Env": sim_env, "Light_Cond": sim_light,
            "Road": sim_road, "Traffic": sim_traffic, "Impact": sim_impact, "Loc_Type": sim_loc,
        }
        risk = predict_risk(model_bundle, scenario)
        tier = risk_tier(risk)
        tier_color = {"Low": "green", "Medium": "orange", "High": "red"}[tier]

        st.markdown("### Predicted Risk")
        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.metric("Injury Probability", f"{risk:.1%}")
            st.markdown(f"**Risk Tier:** :{tier_color}[{tier}]")
        with rc2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk * 100, number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": tier_color},
                       "steps": [{"range": [0, 15], "color": "#d4f4dd"},
                                 {"range": [15, 30], "color": "#ffe8b3"},
                                 {"range": [30, 100], "color": "#ffd2d2"}]},
                title={"text": "Predicted Injury Risk"},
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.caption(f"Model used: **{model_bundle['name']}**, trained on all {len(df):,} 2020 Ottawa collisions.")

st.markdown("---")
st.markdown(
    "**Course Project Note:** This dashboard supports BI analysis, interactive reporting, "
    "and predictive analytics for Ottawa road safety."
)

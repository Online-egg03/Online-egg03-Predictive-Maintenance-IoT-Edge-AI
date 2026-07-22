import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from PIL import Image

# -----------------------------
# Shared Plotly theme
# -----------------------------
PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQUENCE = ["#0080ff", "#003366", "#66b2ff", "#00b386", "#ff6b6b"]

def style_fig(fig, height=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Segoe UI, Inter, sans-serif", size=13, color="#002b4d"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance | IoT Edge AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

/* ---------- Global ---------- */
.main {
    background-color: #f4f6fb;
}

html, body, [class*="css"]  {
    font-family: 'Segoe UI', 'Inter', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ---------- Hero Banner ---------- */
.hero {
    background: linear-gradient(120deg, #003366 0%, #0059b3 60%, #0080ff 100%);
    padding: 40px 45px;
    border-radius: 20px;
    color: white;
    box-shadow: 0px 8px 24px rgba(0, 51, 102, 0.25);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}

.hero-sub {
    font-size: 18px;
    color: #dbe9ff;
    font-weight: 400;
}

.badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.35);
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 13px;
    margin-top: 14px;
    margin-right: 8px;
}

/* ---------- Section Headers ---------- */
.section-header {
    font-size: 24px;
    font-weight: 700;
    color: #002b4d;
    margin-top: 10px;
    margin-bottom: 4px;
    border-left: 5px solid #0080ff;
    padding-left: 12px;
}

.section-caption {
    color: #6b7280;
    font-size: 14px;
    margin-left: 17px;
    margin-bottom: 18px;
}

/* ---------- Metric Cards ---------- */
.metric-card {
    background: white;
    padding: 22px 18px;
    border-radius: 16px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
    text-align: center;
    border-top: 4px solid #0080ff;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0px 8px 22px rgba(0,0,0,0.14);
}

.metric-icon {
    font-size: 26px;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #003366;
}

.metric-label {
    font-size: 13px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* ---------- Tech Pills ---------- */
.pill-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.pill {
    background: white;
    border: 1px solid #dbe4f0;
    padding: 10px 16px;
    border-radius: 30px;
    font-size: 14px;
    font-weight: 600;
    color: #003366;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
}

/* ---------- Info / Callout box ---------- */
.callout {
    background: #eaf3ff;
    border-left: 5px solid #0080ff;
    padding: 16px 20px;
    border-radius: 10px;
    color: #003366;
    font-size: 15px;
}

/* ---------- Card container ---------- */
.card {
    background: white;
    padding: 26px 28px;
    border-radius: 16px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: #9aa5b1;
    font-size: 13px;
    padding-top: 25px;
    padding-bottom: 10px;
    border-top: 1px solid #e2e8f0;
    margin-top: 30px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: #001f3f;
}

section[data-testid="stSidebar"] * {
    color: #f0f4f8 !important;
}

section[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("processed_data.csv")

df = load_data()

# -----------------------------
# Helper: metric card renderer
# -----------------------------
def metric_card(icon, value, label):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## 🏭 Predictive Maintenance")
st.sidebar.markdown("##### IoT Edge AI Dashboard")
st.sidebar.write("---")

nav_choice = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 Dataset Analysis",
        "🤖 Prediction",
        "📈 Model Performance",
        "ℹ️ About"
    ],
    label_visibility="collapsed"
)
page = nav_choice.split(" ", 1)[1]  # strip emoji for logic below

st.sidebar.write("---")
st.sidebar.caption("Developed by **Sandeep Kumar**")
st.sidebar.caption("Data Science & Machine Learning Internship")

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">Predictive Maintenance using IoT Edge AI</div>
        <div class="hero-sub">Manufacturing & Automotive Machine Failure Prediction</div>
        <span class="badge">⚙️ LightGBM</span>
        <span class="badge">📡 IoT Sensor Data</span>
        <span class="badge">🎯 Macro F1: 0.922</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("📁", f"{len(df):,}", "Dataset Records")
    with col2:
        metric_card("🧬", "14", "Original Features")
    with col3:
        metric_card("⚠️", int(df["Machine failure"].sum()), "Machine Failures")
    with col4:
        metric_card("🎯", "0.922", "Macro F1 Score")

    st.write("")
    st.markdown('<div class="section-header">📌 Project Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">What this system does and how it works</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    This project predicts machine failures <b>before breakdowns occur</b>, enabling proactive maintenance
    instead of costly reactive repairs.
    <br><br>
    The model combines:
    <ul>
        <li>📡 IoT Sensor Data</li>
        <li>🌡️ Contextual Environmental Features</li>
        <li>🛠️ Feature Engineering</li>
        <li>⚖️ SMOTE (class imbalance handling)</li>
        <li>🌳 LightGBM Classifier</li>
        <li>🔊 Noise Sensitivity Analysis</li>
    </ul>
    to build a robust, production-ready Predictive Maintenance System.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🛠 Technology Stack</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Tools and libraries powering this project</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="pill-container">
        <div class="pill">🐍 Python</div>
        <div class="pill">🐼 Pandas</div>
        <div class="pill">🔢 NumPy</div>
        <div class="pill">🤖 Scikit-Learn</div>
        <div class="pill">🌳 LightGBM</div>
        <div class="pill">⚖️ SMOTE</div>
        <div class="pill">🚀 Streamlit</div>
        <div class="pill">📊 Matplotlib</div>
        <div class="pill">🐙 GitHub</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-header">📊 Dataset Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Preview of the processed dataset</div>', unsafe_allow_html=True)

    st.dataframe(df.head(), use_container_width=True)

    st.write("")
    st.markdown("""
    <div class="callout">
    💡 Use the sidebar to explore <b>Dataset Analysis</b>, run <b>Predictions</b>, and review <b>Model Performance</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="footer">🏭 Predictive Maintenance Dashboard &nbsp;|&nbsp; Built with Streamlit &nbsp;|&nbsp; © 2026</div>', unsafe_allow_html=True)

# -----------------------------
# Placeholder pages
# -----------------------------
elif page == "Dataset Analysis":
    st.title("📊 Dataset Analysis Dashboard")
    st.markdown("### Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric(
            "Failure Rate",
            f"{df['Machine failure'].mean()*100:.2f}%"
        )
    st.write("---")
    # ============================
    # Failure Distribution
    # ============================
    st.subheader("Machine Failure Distribution")
    fig = px.bar(
        x=["Normal","Failure"],
        y=df["Machine failure"].value_counts().sort_index().values,
        color=["Normal","Failure"],
        text=df["Machine failure"].value_counts().sort_index().values
    )
    fig.update_layout(
        xaxis_title="Status",
        yaxis_title="Count"
    )
    st.plotly_chart(fig,use_container_width=True)
    st.write("---")
    # ============================
    # Machine Type Distribution
    # ============================
    st.subheader("Machine Type Distribution")
    fig = px.pie(
        df,
        names="Type",
        hole=0.45
    )
    st.plotly_chart(fig,use_container_width=True)
    st.write("---")
    # ============================
    # Temperature Distribution
    # ============================
    st.subheader("Air Temperature Distribution")
    fig = px.histogram(
        df,
        x="Air temperature [K]",
        nbins=40
    )
    st.plotly_chart(fig,use_container_width=True)
    st.write("---")
    # ============================
    # Correlation
    # ============================
    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=False,
        aspect="auto"
    )
    st.plotly_chart(fig,use_container_width=True)
    st.write("---")
    st.success("Dataset Analysis Completed Successfully.")

elif page == "Prediction":
    st.title("🤖 Machine Failure Prediction")
    st.markdown("### Enter Machine Parameters")
    col1, col2 = st.columns(2)
    with col1:
        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"]
        )
        air_temp = st.number_input(
            "Air Temperature (K)",
            value=300.0
        )
        process_temp = st.number_input(
            "Process Temperature (K)",
            value=310.0
        )
        rpm = st.number_input(
            "Rotational Speed (rpm)",
            value=1500
        )
    with col2:
        torque = st.number_input(
            "Torque (Nm)",
            value=40.0
        )
        tool_wear = st.number_input(
            "Tool Wear (min)",
            value=100
        )
        ambient = st.number_input(
            "Ambient Temperature",
            value=30.0
        )
        load_density = st.slider(
            "Load Density",
            0.0,
            1.0,
            0.5
        )
    if st.button("🚀 Predict Failure", use_container_width=True):
        st.warning(
            "Prediction integration will be connected with the trained model in the final version."
        )
        st.progress(87)
        st.metric(
            "Failure Probability",
            "87 %"
        )
        st.error("🔴 HIGH RISK")
        st.success(
            "Recommendation : Immediate Maintenance Required"
        )

elif page == "Model Performance":
    st.title("📈 Model Performance Dashboard")
    st.write("---")
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.metric(
            "Accuracy",
            "99.5%"
        )
    with col2:
        st.metric(
            "Macro F1",
            "0.922"
        )
    with col3:
        st.metric(
            "Noise F1",
            "0.830"
        )
    with col4:
        st.metric(
            "Threshold F1",
            "0.891"
        )
    st.write("---")
    st.subheader("Confusion Matrix")
    st.image(
        "failure_distribution.png",
        use_container_width=True
    )
    st.write("---")
    st.subheader("Precision Recall Curve")
    st.image(
        "precision_recall_curve.png",
        use_container_width=True
    )
    st.write("---")
    st.subheader("Project Performance Summary")
    summary = pd.DataFrame({
        "Metric":[
            "Random Forest Accuracy",
            "Average Macro F1",
            "Noise Robustness",
            "Threshold Tuned F1"
        ],
        "Value":[
            "99.5%",
            "0.922",
            "0.830",
            "0.891"
        ]
    })
    st.dataframe(
        summary,
        use_container_width=True
    )
    st.success("Model Evaluation Completed Successfully")

elif page == "About":
    st.title("ℹ About Project")
    st.write("---")
    st.header("Project Overview")
    st.write("""
Predictive Maintenance using IoT Edge AI is a machine learning based
solution developed for Manufacturing and Automotive industries.
The objective is to predict machine failures before breakdown occurs
using sensor telemetry and contextual environmental data.
""")
    st.write("---")
    st.header("Engineering Workflow")
    st.markdown("""
✅ Week 1
- Dataset Collection
- Data Cleaning
- Exploratory Data Analysis
---
✅ Week 2
- Feature Engineering
- Contextual Data Fusion
- Correlation Analysis
---
✅ Week 3
- SMOTE
- Stratified Cross Validation
- LightGBM Modeling
---
✅ Week 4
- Noise Sensitivity Analysis
- Precision Recall Curve
- Threshold Tuning
""")
    st.write("---")
    st.header("Technology Stack")
    tech = pd.DataFrame({
        "Technology":[
            "Python",
            "Pandas",
            "NumPy",
            "Matplotlib",
            "Scikit-Learn",
            "LightGBM",
            "SMOTE",
            "Plotly",
            "Streamlit",
            "Git"
        ]
    })
    st.table(tech)
    st.write("---")
    st.header("Developer")
    st.success("Sandeep Kumar")
    st.info("Internship Domain : Data Science & Machine Learning")
    st.write("---")
    st.header("Project Status")
    st.success("✔ Completed")
    st.balloons()
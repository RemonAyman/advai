import sys
import os

# Auto-launcher: If executed with `python app.py`, automatically boots the Streamlit server
try:
    from streamlit.runtime import exists
    if not exists():
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())
except ImportError:
    pass

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(
    page_title="Breast Cancer CAD AI System | Eng Mathew",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Session State Initialization
if "started" not in st.session_state:
    st.session_state.started = False

# 3. Custom CSS Styling (Dark & Light Mode Compatible, Sleek Medical Tech Theme)
st.markdown("""
<style>
    /* Global font and smooth design */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Welcome Hero Container */
    .welcome-container {
        background: linear-gradient(135deg, rgba(14, 116, 144, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid rgba(14, 116, 144, 0.3);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.15);
    }
    .welcome-badge {
        display: inline-block;
        background: linear-gradient(90deg, #0284c7, #6366f1);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }
    .welcome-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
    }
    .welcome-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        max-width: 800px;
        margin: 0 auto 1.8rem auto;
        line-height: 1.6;
    }

    /* Stat Cards */
    .stat-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 1.4rem 1rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        border-color: #38bdf8;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 0.2rem;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Diagnosis Result Badges */
    .benign-badge {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        color: #10b981;
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px solid #10b981;
        font-size: 1.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.2);
    }
    .malignant-badge {
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.15) 0%, rgba(190, 18, 60, 0.25) 100%);
        color: #f43f5e;
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px solid #e11d48;
        font-size: 1.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(225, 29, 72, 0.2);
    }

    /* Team Pill */
    .team-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 30px;
        padding: 0.6rem 1.4rem;
        margin: 0.3rem;
        font-size: 0.9rem;
        color: #cbd5e1;
    }

    /* Clean navigation header */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 0.8rem 1.4rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 4. Model & Resource Loader
@st.cache_resource
def load_resources():
    imputer = joblib.load('models/imputer.joblib')
    scaler = joblib.load('models/scaler.joblib')

    available_models = {}
    model_files = {
        'Random Forest (Recommended)': 'models/random_forest.joblib',
        'Support Vector Machine (SVM)': 'models/support_vector_machine_svm.joblib',
        'Artificial Neural Network (ANN)': 'models/artificial_neural_network_ann.joblib',
        'Voting Ensemble (RF+ANN+SVM)': 'models/ensemble_voting.joblib',
        'K-Nearest Neighbors (KNN)': 'models/knn.joblib',
        'Decision Tree (DT)': 'models/decision_tree.joblib'
    }
    for label, path in model_files.items():
        if os.path.exists(path):
            available_models[label] = joblib.load(path)

    with open('models/metadata.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)

    return imputer, scaler, available_models, meta

try:
    imputer, scaler, models_dict, meta = load_resources()
except Exception as e:
    st.error(f"Error loading models or metadata. Please ensure models exist in `models/`. Details: {e}")
    st.stop()


# ==============================================================================
# SCREEN 1: WELCOME & LANDING SCREEN (Hello Eng Mathew)
# ==============================================================================
if not st.session_state.started:
    # Hero Welcome Box
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-badge">🎗️ Advanced AI Project — Supervised Learning Classification</div>
        <div class="welcome-title">Hello Eng Mathew 👋</div>
        <div class="welcome-subtitle">
            Welcome to our <b>Breast Cancer Cytology Classification & CAD Diagnostic System</b>.<br>
            An end-to-end clinical decision-support platform designed to assist cytopathologists and oncologists in early malignancy detection using fine-needle aspirate (FNA) cellular morphometry.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Quick Highlights
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">97.12%</div>
            <div class="stat-label">🏆 Peak Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">95.83%</div>
            <div class="stat-label">🎯 Clinical Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">0.9961</div>
            <div class="stat-label">📈 Highest ROC-AUC</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">6 Models</div>
            <div class="stat-label">⚡ ML + Ensemble</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Central Call To Action
    start_col1, start_col2, start_col3 = st.columns([1, 1.6, 1])
    with start_col2:
        st.info("💡 **Ready to explore the diagnostic models?** Click below to launch the interactive live patient testing environment.")
        if st.button("🚀 START DIAGNOSTIC SYSTEM", type="primary", use_container_width=True):
            st.session_state.started = True
            st.rerun()

    st.write("")
    st.markdown("---")

    # Team Members & Architecture Preview
    st.subheader("👥 Project Development Team")
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <span class="team-pill">👨‍💻 Remon Ayman</span>
        <span class="team-pill">👨‍💻 Marcelino</span>
        <span class="team-pill">👨‍💻 Bishoy</span>
        <span class="team-pill">👨‍💻 Mario</span>
        <span class="team-pill">👩‍💻 Engy</span>
        <span class="team-pill">👩‍💻 Karen</span>
        <span class="team-pill">👨‍💻 David</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📌 Project Architecture & Dataset Overview (Click to Preview)"):
        st.markdown("""
        - **Dataset**: UCI Breast Cancer Wisconsin (Original) by Dr. William H. Wolberg (699 patients, 9 cytological features).
        - **Data Cleaning**: Handled 16 missing values in `Bare_Nuclei` with strictly-fit median imputation; removed 8 exact duplicate records (691 clean instances).
        - **Stratified Split**: 80% Training (552 patients) : 20% Testing (139 patients) preserving exact 34.5% malignancy ratio.
        - **Preprocessing**: Leak-free `StandardScaler` fitted strictly on training data.
        - **Trained Algorithms**: Random Forest, Artificial Neural Network (ANN), Support Vector Machine (SVM), Voting Ensemble, K-Nearest Neighbors (KNN), and Decision Tree.
        """)

# ==============================================================================
# SCREEN 2: MAIN INTERACTIVE DIAGNOSTIC SYSTEM
# ==============================================================================
else:
    # Top Bar: Personalized Navigation
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 5px;">
            <span style="font-size: 1.6rem; font-weight: 800; color: #38bdf8;">🎗️ CAD Diagnostic Platform</span>
            <span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 3px 10px; border-radius: 20px; font-size: 0.85rem; font-weight: 700;">
                Session: Eng Mathew
            </span>
        </div>
        """, unsafe_allow_html=True)
    with top_col2:
        if st.button("🏠 Return to Welcome", use_container_width=True):
            st.session_state.started = False
            st.rerun()

    # Sidebar: Model and Presets
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 0.5rem; background: rgba(56, 189, 248, 0.1); border-radius: 12px; margin-bottom: 1rem;">
        <span style="font-size: 1.1rem; font-weight: 700; color: #38bdf8;">Welcome, Eng Mathew! 👋</span><br>
        <span style="font-size: 0.8rem; color: #94a3b8;">Diagnostic Control Panel</span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("⚙️ Model Configuration")
    selected_model_name = st.sidebar.selectbox(
        "Select Classification Model:",
        options=list(models_dict.keys()),
        index=0
    )
    active_model = models_dict[selected_model_name]

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Quick Clinical Presets")
    preset = st.sidebar.radio(
        "Populate Biopsy Attributes:",
        options=["Custom Manual Entry", "Typical Benign Patient", "Typical Malignant Patient", "Borderline / Equivocal Case"]
    )

    # Preset values
    default_values = {
        'Clump_Thickness': 5,
        'Uniformity_Cell_Size': 1,
        'Uniformity_Cell_Shape': 1,
        'Marginal_Adhesion': 1,
        'Single_Epithelial_Cell_Size': 2,
        'Bare_Nuclei': 1,
        'Bland_Chromatin': 3,
        'Normal_Nucleoli': 1,
        'Mitoses': 1
    }

    if preset == "Typical Benign Patient":
        default_values = {k: 1 for k in default_values}
        default_values['Clump_Thickness'] = 2
        default_values['Single_Epithelial_Cell_Size'] = 2
    elif preset == "Typical Malignant Patient":
        default_values = {
            'Clump_Thickness': 8,
            'Uniformity_Cell_Size': 9,
            'Uniformity_Cell_Shape': 8,
            'Marginal_Adhesion': 7,
            'Single_Epithelial_Cell_Size': 8,
            'Bare_Nuclei': 10,
            'Bland_Chromatin': 7,
            'Normal_Nucleoli': 8,
            'Mitoses': 5
        }
    elif preset == "Borderline / Equivocal Case":
        default_values = {
            'Clump_Thickness': 5,
            'Uniformity_Cell_Size': 4,
            'Uniformity_Cell_Shape': 4,
            'Marginal_Adhesion': 3,
            'Single_Epithelial_Cell_Size': 4,
            'Bare_Nuclei': 3,
            'Bland_Chromatin': 4,
            'Normal_Nucleoli': 3,
            'Mitoses': 2
        }

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩺 Live Patient Diagnosis",
        "📊 Model Benchmark & Comparison",
        "📖 Cytological Features Guide",
        "ℹ️ Project Info & Architecture"
    ])

    # ---------------- TAB 1: LIVE DIAGNOSIS ----------------
    with tab1:
        st.markdown("### Enter Patient Biopsy Cytological Measurements (Scale: 1 - 10)")
        st.caption("Graded from **1 (Normal / Healthy)** to **10 (Most Severe Abnormality)** based on Fine Needle Aspirate (FNA) cellular examination.")

        col1, col2, col3 = st.columns(3)

        with col1:
            clump = st.slider("1. Clump Thickness", 1, 10, default_values['Clump_Thickness'], help="Degree of multi-layered cellular agglomeration.")
            u_size = st.slider("2. Uniformity of Cell Size", 1, 10, default_values['Uniformity_Cell_Size'], help="Variability in cell dimensions (classic hallmark of cancer).")
            u_shape = st.slider("3. Uniformity of Cell Shape", 1, 10, default_values['Uniformity_Cell_Shape'], help="Variability in cell boundary geometry and irregular contours.")

        with col2:
            adhesion = st.slider("4. Marginal Adhesion", 1, 10, default_values['Marginal_Adhesion'], help="Tendency of cells to stick together; loss of adhesion enables metastasis.")
            epi_size = st.slider("5. Single Epithelial Cell Size", 1, 10, default_values['Single_Epithelial_Cell_Size'], help="Epithelial cell volume and nuclear-to-cytoplasmic ratio.")
            bare_nuc = st.slider("6. Bare Nuclei", 1, 10, default_values['Bare_Nuclei'], help="Nuclei without surrounding cytoplasm (strong indicator of malignancy).")

        with col3:
            bland_chr = st.slider("7. Bland Chromatin", 1, 10, default_values['Bland_Chromatin'], help="Coarseness and aggregation of nuclear chromatin.")
            norm_nuc = st.slider("8. Normal Nucleoli", 1, 10, default_values['Normal_Nucleoli'], help="Prominence and hyperactivity of nucleoli.")
            mitoses = st.slider("9. Mitoses", 1, 10, default_values['Mitoses'], help="Frequency of cells undergoing active mitotic division.")

        feature_dict = {
            'Clump_Thickness': clump,
            'Uniformity_Cell_Size': u_size,
            'Uniformity_Cell_Shape': u_shape,
            'Marginal_Adhesion': adhesion,
            'Single_Epithelial_Cell_Size': epi_size,
            'Bare_Nuclei': bare_nuc,
            'Bland_Chromatin': bland_chr,
            'Normal_Nucleoli': norm_nuc,
            'Mitoses': mitoses
        }

        patient_df = pd.DataFrame([feature_dict])

        st.markdown("---")
        predict_btn = st.button("🔬 Analyze Biopsy & Run Diagnosis", type="primary", use_container_width=True)

        if predict_btn or preset != "Custom Manual Entry":
            # Leak-free preprocessing
            patient_imp = pd.DataFrame(imputer.transform(patient_df), columns=list(feature_dict.keys()))
            patient_scaled = pd.DataFrame(scaler.transform(patient_imp), columns=list(feature_dict.keys()))

            prediction = active_model.predict(patient_scaled)[0]
            probabilities = active_model.predict_proba(patient_scaled)[0]

            prob_benign = probabilities[0] * 100
            prob_malignant = probabilities[1] * 100

            st.write("")
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                st.subheader("Diagnostic Classification:")
                if prediction == 1:
                    st.markdown(f"""
                    <div class="malignant-badge">
                        ⚠️ DIAGNOSIS: MALIGNANT (ورم سرطاني خبيث)<br>
                        <span style="font-size: 1.1rem; font-weight: 500;">
                            Malignancy Probability: <b>{prob_malignant:.2f}%</b> (Confidence: High)
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.error("🚨 **Clinical Recommendation**: Immediate oncology referral, histopathological confirmation, and staging investigation advised.")
                else:
                    st.markdown(f"""
                    <div class="benign-badge">
                        ✅ DIAGNOSIS: BENIGN (ورم حميد سليم)<br>
                        <span style="font-size: 1.1rem; font-weight: 500;">
                            Benign Probability: <b>{prob_benign:.2f}%</b> (Confidence: High)
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("✔️ **Clinical Recommendation**: Low suspicion of malignancy. Recommend routine surveillance and follow-up screening.")

                st.markdown("#### Probability Meter:")
                st.progress(prob_malignant / 100)
                pcol1, pcol2 = st.columns(2)
                pcol1.metric("Benign Probability", f"{prob_benign:.1f}%")
                pcol2.metric("Malignant Probability", f"{prob_malignant:.1f}%")

            with res_col2:
                st.subheader("Cellular Abnormality Profile:")
                fig, ax = plt.subplots(figsize=(6, 4))
                feats = [f.replace('_', ' ') for f in feature_dict.keys()]
                vals = list(feature_dict.values())

                y_pos = np.arange(len(feats))
                bar_color = '#f43f5e' if prediction == 1 else '#38bdf8'
                ax.barh(y_pos, vals, color=bar_color, alpha=0.85)
                ax.axvline(1, color='#10b981', linestyle='--', linewidth=1.5, label='Healthy Baseline (1.0)')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feats, fontsize=8)
                ax.set_xlim(0, 10.5)
                ax.set_xlabel('Cytological Rating (1 - 10)', fontsize=9)
                ax.set_title('Feature Severity vs Normal Cell Baseline', fontsize=10, fontweight='bold')
                ax.legend(loc='lower right', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)

    # ---------------- TAB 2: MODEL BENCHMARK ----------------
    with tab2:
        st.subheader("📊 Model Performance Benchmark & Comparison")
        st.write("All models were rigorously evaluated on the held-out test set ($N=139$) using **5-Fold Stratified Cross-Validation**.")

        results_table = pd.read_csv('model_comparison_results.csv')

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("🏆 Peak Accuracy", "97.12%", "Random Forest / ANN / SVM")
        m_col2.metric("🎯 Peak Recall (Sensitivity)", "95.83%", "Only 2 False Negatives")
        m_col3.metric("📈 Highest ROC-AUC", "0.9961", "Artificial Neural Network")

        display_df = results_table.copy()
        display_df['Accuracy'] = display_df['Accuracy'].map(lambda x: f"{float(x)*100:.2f}%")
        display_df['Precision'] = display_df['Precision'].map(lambda x: f"{float(x)*100:.2f}%")
        display_df['Recall'] = display_df['Recall'].map(lambda x: f"{float(x)*100:.2f}%")
        display_df['F1-Score'] = display_df['F1-Score'].map(lambda x: f"{float(x):.4f}")
        display_df['ROC-AUC'] = display_df['ROC-AUC'].map(lambda x: f"{float(x):.4f}")

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if os.path.exists("figures/7_all_confusion_matrices.png"):
                st.image("figures/7_all_confusion_matrices.png", caption="Confusion Matrices Across All Models (Test Set N=139)")
        with col_img2:
            if os.path.exists("figures/8_roc_curves_comparison.png"):
                st.image("figures/8_roc_curves_comparison.png", caption="Receiver Operating Characteristic (ROC) Curves")

        if os.path.exists("figures/10_rf_feature_importance.png"):
            st.image("figures/10_rf_feature_importance.png", caption="Random Forest Gini Feature Importance Analysis")

    # ---------------- TAB 3: CYTOLOGICAL CRITERIA ----------------
    with tab3:
        st.subheader("📖 Cytological Morphometry Guide (Dr. William H. Wolberg)")
        st.markdown("""
        | # | Cytological Feature | Clinical Description | Pathological Significance |
        |---|---|---|---|
        | 1 | **Clump Thickness** | Grouping of cells in multi-layers vs monolayers | Benign cells form flat monolayers; cancer cells agglomerate in thick mounds. |
        | 2 | **Uniformity of Cell Size** | Variation in size among biopsied cells | High variance (anisonucleosis) is a primary hallmark of malignant transformation. |
        | 3 | **Uniformity of Cell Shape** | Structural irregularity in cell membranes | Cancerous cells display pleomorphism and bizarre geometric shapes. |
        | 4 | **Marginal Adhesion** | Strength of intercellular bonds | Loss of adhesion facilitates local tissue invasion and vascular metastasis. |
        | 5 | **Single Epithelial Size** | Cytoplasmic enlargement & nuclear ratio | High values indicate malignant cellular hypertrophy. |
        | 6 | **Bare Nuclei** | Nuclei stripped of their cytoplasmic boundary | Frequently found in malignant aspirates; powerful predictor of high-grade lesions. |
        | 7 | **Bland Chromatin** | Texture of nuclear chromatin | Fine chromatin is normal; coarse, clumped chromatin marks cancer. |
        | 8 | **Normal Nucleoli** | Presence and prominence of nucleoli | Multiple, hypertrophic nucleoli reflect rampant ribosomal synthesis in tumor cells. |
        | 9 | **Mitoses** | Number of active cell divisions observed | High mitotic count correlates directly with aggressive histological proliferation. |
        """)

    # ---------------- TAB 4: PROJECT INFO ----------------
    with tab4:
        st.subheader("ℹ️ Project Architecture & Pipeline Summary")
        st.markdown("""
        - **Supervised Learning Objective**: Binary classification of cytological biopsies into Benign (0) vs. Malignant (1).
        - **Dataset**: UCI Machine Learning Repository — Breast Cancer Wisconsin (Original).
        - **Preprocessing Protocol**: Strictly leak-free (median imputation & `StandardScaler` fitted only on training set).
        - **Evaluation Protocol**: 5-Fold Stratified Cross-Validation on training folds + held-out test evaluation ($N=139$).
        - **Champion Model**: **Random Forest** selected for peak clinical recall (95.83%), peak accuracy (97.12%), and full feature explainability.
        - **Session Recipient**: Dedicated review for **Eng Mathew**.
        - **GitHub Source**: [https://github.com/RemonAyman/advai](https://github.com/RemonAyman/advai)
        """)

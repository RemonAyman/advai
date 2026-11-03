import sys
import os

# Auto-launcher: If executed with `python app.py` (e.g., Run button in PyCharm / VS Code),
# automatically boot the Streamlit web server so the application launches in the browser!
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
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="Breast Cancer CAD Diagnostic System",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1A365D;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #0E7490;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .benign-badge {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 8px solid #10B981;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
    }
    .malignant-badge {
        background-color: #FFE4E6;
        color: #9F1239;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 8px solid #E11D48;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Helper to load models
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
    st.error(f"Error loading models or metadata. Please ensure `train_models_extensive.py` has finished. Details: {e}")
    st.stop()

# Title
st.markdown('<div class="main-header">🎗️ Breast Cancer Cytology Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Supervised Learning Second-Opinion Diagnostic Tool | UCI Breast Cancer Wisconsin</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Model & Diagnostic Settings")
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

with tab1:
    st.markdown("### Enter Patient Biopsy Cytological Measurements (Scale: 1 - 10)")
    st.write("Graded from **1 (Normal / Healthy)** to **10 (Most Severe Abnormality)** based on Fine Needle Aspirate (FNA) examination.")

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
    predict_btn = st.button("🔬 Analyze Biopsy & Predict Diagnosis", type="primary")

    if predict_btn or preset != "Custom Manual Entry":
        # Preprocess patient data
        patient_imp = pd.DataFrame(imputer.transform(patient_df), columns=list(feature_dict.keys()))
        patient_scaled = pd.DataFrame(scaler.transform(patient_imp), columns=list(feature_dict.keys()))

        prediction = active_model.predict(patient_scaled)[0]
        probabilities = active_model.predict_proba(patient_scaled)[0]

        prob_benign = probabilities[0] * 100
        prob_malignant = probabilities[1] * 100

        st.subheader("Diagnostic Prediction Result:")
        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="malignant-badge">
                    ⚠️ DIAGNOSIS: MALIGNANT (ورم خبيث)<br>
                    <span style="font-size: 1.1rem; font-weight: normal;">
                        Malignancy Probability: <b>{prob_malignant:.2f}%</b> (Confidence: High)
                    </span>
                </div>
                """, unsafe_allow_html=True)
                st.error("🚨 Clinical Recommendation: Immediate pathological review, surgical oncology consultation, and confirmatory histopathology required.")
            else:
                st.markdown(f"""
                <div class="benign-badge">
                    ✅ DIAGNOSIS: BENIGN (ورم حميد)<br>
                    <span style="font-size: 1.1rem; font-weight: normal;">
                        Benign Probability: <b>{prob_benign:.2f}%</b> (Confidence: High)
                    </span>
                </div>
                """, unsafe_allow_html=True)
                st.success("✔️ Clinical Recommendation: Low risk of malignancy. Recommend routine standard mammographic / ultrasound follow-up.")

            # Metric probability meters
            st.markdown("#### Probability Distribution:")
            st.progress(prob_malignant / 100)
            pcol1, pcol2 = st.columns(2)
            pcol1.metric("Benign Probability", f"{prob_benign:.1f}%")
            pcol2.metric("Malignant Probability", f"{prob_malignant:.1f}%")

        with res_col2:
            st.markdown("#### Patient Feature Profile vs. Healthy Baseline:")
            fig, ax = plt.subplots(figsize=(6, 4))
            feats = [f.replace('_', ' ') for f in feature_dict.keys()]
            vals = list(feature_dict.values())
            
            y_pos = np.arange(len(feats))
            ax.barh(y_pos, vals, color='#e11d48' if prediction==1 else '#0e7490', alpha=0.85)
            ax.axvline(1, color='green', linestyle='--', label='Normal Healthy Baseline (Score=1)')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feats, fontsize=8)
            ax.set_xlim(0, 10.5)
            ax.set_xlabel('Cytological Rating (1-10)', fontsize=9)
            ax.set_title('Biopsy Severity Spectrum', fontsize=10, fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)

with tab2:
    st.subheader("📊 Model Performance Benchmark & Comparison")
    st.write("All models were rigorously evaluated on the held-out test set ($N=139$) using 5-Fold Stratified Cross-Validation.")

    results_table = pd.read_csv('model_comparison_results.csv')
    st.dataframe(results_table.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='#D1FAE5'))

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists("figures/7_all_confusion_matrices.png"):
            st.image("figures/7_all_confusion_matrices.png", caption="Confusion Matrices across all 5 models (Test Set N=139)")
    with col_img2:
        if os.path.exists("figures/8_roc_curves_comparison.png"):
            st.image("figures/8_roc_curves_comparison.png", caption="Receiver Operating Characteristic (ROC) Curves")

    if os.path.exists("figures/10_rf_feature_importance.png"):
        st.image("figures/10_rf_feature_importance.png", caption="Random Forest Feature Importance Analysis")

with tab3:
    st.subheader("📖 Cytological Criteria & Grading Guide (Dr. William H. Wolberg)")
    st.markdown("""
    | # | Feature | Clinical Description | Pathological Significance |
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

with tab4:
    st.subheader("ℹ️ Project Architecture & Machine Learning Pipeline")
    st.markdown("""
    - **Dataset**: UCI Machine Learning Repository — Breast Cancer Wisconsin (Original)
    - **Preprocessing Guarantee**: Strictly leak-free preprocessing (imputation and standard scaling fit exclusively on training folds).
    - **Cross-Validation**: 5-Fold Stratified Cross-Validation ensuring invariant class distributions across folds.
    - **Champion Model**: **Random Forest** selected for optimal blend of peak accuracy (97.12%), clinical recall (95.83%), and transparent interpretability for medical practitioners.
    - **GitHub Repository**: [RemonAyman/advai](https://github.com/RemonAyman/advai)
    """)

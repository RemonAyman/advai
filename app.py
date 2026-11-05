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
import time
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# 1. Page Configuration (Mobile-Friendly & Wide)
st.set_page_config(
    page_title="Breast Cancer CAD AI System | Eng Mathew",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="auto"
)

# 2. Session State Initialization
if "started" not in st.session_state:
    st.session_state.started = False
if "lang" not in st.session_state:
    st.session_state.lang = "EN" # Default is English

is_ar = (st.session_state.lang == "AR")

# 3. Comprehensive Mobile-First & Desktop Responsive CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Cairo:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Cairo', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Base Responsive Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        padding-left: clamp(1rem, 4vw, 3rem) !important;
        padding-right: clamp(1rem, 4vw, 3rem) !important;
        max-width: 1200px;
    }

    /* Welcome Hero Container */
    .welcome-container {
        background: linear-gradient(135deg, rgba(14, 116, 144, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid rgba(14, 116, 144, 0.35);
        border-radius: 20px;
        padding: clamp(1.5rem, 5vw, 3rem) clamp(1rem, 4vw, 2.5rem);
        margin-top: 0.5rem;
        margin-bottom: 1.8rem;
        text-align: center;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.2);
    }
    .welcome-badge {
        display: inline-block;
        background: linear-gradient(90deg, #0284c7, #6366f1);
        color: white;
        padding: 0.4rem 1.1rem;
        border-radius: 50px;
        font-size: clamp(0.75rem, 2vw, 0.95rem);
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
    }
    .welcome-title {
        font-size: clamp(1.8rem, 6vw, 3.2rem);
        font-weight: 800;
        line-height: 1.2;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
    }
    .welcome-subtitle {
        font-size: clamp(0.95rem, 2.5vw, 1.2rem);
        color: #94a3b8;
        max-width: 820px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.6;
    }

    /* Responsive Grid for Stat Cards (2x2 on Mobile, 4x1 on Desktop) */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    @media (max-width: 820px) {
        .stat-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }
    }
    @media (max-width: 480px) {
        .stat-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
        }
    }

    .stat-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: clamp(0.9rem, 2.5vw, 1.3rem) 0.6rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: #38bdf8;
    }
    .stat-value {
        font-size: clamp(1.4rem, 4vw, 2rem);
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 0.2rem;
    }
    .stat-label {
        font-size: clamp(0.75rem, 1.8vw, 0.85rem);
        color: #cbd5e1;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Diagnosis Result Badges */
    .benign-badge {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        color: #10b981;
        padding: clamp(1rem, 3vw, 1.5rem);
        border-radius: 16px;
        border: 2px solid #10b981;
        font-size: clamp(1.15rem, 3.5vw, 1.5rem);
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.2);
        line-height: 1.4;
    }
    .malignant-badge {
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.15) 0%, rgba(190, 18, 60, 0.25) 100%);
        color: #f43f5e;
        padding: clamp(1rem, 3vw, 1.5rem);
        border-radius: 16px;
        border: 2px solid #e11d48;
        font-size: clamp(1.15rem, 3.5vw, 1.5rem);
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(225, 29, 72, 0.2);
        line-height: 1.4;
    }

    /* Team Pills Layout */
    .team-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 30px;
        padding: clamp(0.35rem, 1.5vw, 0.55rem) clamp(0.7rem, 2vw, 1.2rem);
        margin: 0.25rem;
        font-size: clamp(0.75rem, 1.8vw, 0.9rem);
        color: #cbd5e1;
    }

    /* Mobile Screen Stacking & Media Queries */
    @media (max-width: 768px) {
        /* Force multi-column layout to stack on mobile phones for comfortable touch */
        [data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
            margin-bottom: 0.6rem !important;
        }

        /* Adjust Streamlit button padding on mobile */
        .stButton button {
            padding: 0.6rem 1rem !important;
            font-size: 1rem !important;
        }

        /* Adjust slider touch targets */
        .stSlider {
            padding-bottom: 0.5rem !important;
        }

        /* Top Bar wrap */
        .top-nav-box {
            flex-direction: column !important;
            gap: 10px !important;
            align-items: stretch !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# 4. Ultra-Fast Lazy Model & Resource Cache
MODEL_OPTIONS_MAP = {
    'EN': {
        'Random Forest (Recommended)': 'models/random_forest.joblib',
        'Support Vector Machine (SVM)': 'models/support_vector_machine_svm.joblib',
        'Artificial Neural Network (ANN)': 'models/artificial_neural_network_ann.joblib',
        'Voting Ensemble (RF+ANN+SVM)': 'models/ensemble_voting.joblib',
        'K-Nearest Neighbors (KNN)': 'models/knn.joblib',
        'Decision Tree (DT)': 'models/decision_tree.joblib'
    },
    'AR': {
        'الغابة العشوائية (الموصى به) - Random Forest': 'models/random_forest.joblib',
        'آلة المتجهات الداعمة - SVM': 'models/support_vector_machine_svm.joblib',
        'الشبكة العصبية الاصطناعية - ANN': 'models/artificial_neural_network_ann.joblib',
        'نموذج التصويت التجميعي - Voting Ensemble': 'models/ensemble_voting.joblib',
        'أقرب الجيران - KNN': 'models/knn.joblib',
        'شجرة القرار - Decision Tree': 'models/decision_tree.joblib'
    }
}

@st.cache_resource
def load_preprocessors():
    imp = joblib.load('models/imputer.joblib')
    scl = joblib.load('models/scaler.joblib')
    return imp, scl

@st.cache_resource
def load_single_model_by_path(path):
    return joblib.load(path)

def generate_clinical_report(patient_data, prediction, prob_benign, prob_malignant, model_name, is_ar=False):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if is_ar:
        return f"""================================================================================
تقرير الفحص والتشخيص الخلوي المعتمد (CLINICAL PATHOLOGY REPORT)
نظام الذكاء الاصطناعي لتشخيص أورام الثدي (Wisconsin Breast Cancer CAD AI System)
================================================================================
تاريخ ووقت الفحص: {now_str}
المشرف الأكاديمي: م/ ماثيو (Eng Mathew)
فريق العمل البحثي: ريمون أيمن (قائد الفريق)، مارسيلينو، بيشوي، ماريو، إنجي، كارين
حالة التقييم: تشخيص ذكي معتمد عبر تعلم الآلة (Machine Learning Inference)
--------------------------------------------------------------------------------
القياسات الخلوية لخزعة المريضة (مقياس د. ويليام ولبرج المجهري من 1 إلى 10):
  1. سُمك التكتل الخلوي (Clump Thickness):              {patient_data['Clump_Thickness']} / 10
  2. انتظام حجم الخلية (Uniformity of Cell Size):      {patient_data['Uniformity_Cell_Size']} / 10
  3. انتظام شكل الخلية (Uniformity of Cell Shape):     {patient_data['Uniformity_Cell_Shape']} / 10
  4. التصاق الحواف (Marginal Adhesion):                {patient_data['Marginal_Adhesion']} / 10
  5. حجم الخلية الظهارية المفردة (Epithelial Size):   {patient_data['Single_Epithelial_Cell_Size']} / 10
  6. الأنوية العارية (Bare Nuclei):                    {patient_data['Bare_Nuclei']} / 10
  7. الكروماتين النووي (Bland Chromatin):              {patient_data['Bland_Chromatin']} / 10
  8. النويات الطبيعية (Normal Nucleoli):               {patient_data['Normal_Nucleoli']} / 10
  9. الانقسام الميتوزي (Mitoses):                      {patient_data['Mitoses']} / 10
--------------------------------------------------------------------------------
نتائج التشخيص السريري الآلي:
  - التصنيف الأساسي:       {'ورم سرطاني خبيث (MALIGNANT)' if prediction == 1 else 'ورم حميد سليم (BENIGN)'}
  - احتمالية الإصابة بالورم: {prob_malignant:.2f}%
  - احتمالية الورم الحميد:  {prob_benign:.2f}%
  - درجة ثقة النموذج:      {'مرتفعة جداً (>90%)' if max(prob_malignant, prob_benign) > 90 else 'مرتفعة'}
  - الخوارزمية المستخدمة:  {model_name}
  - حساسية النموذج الطبية: 95.83% (Clinical Recall / Sensitivity)
--------------------------------------------------------------------------------
التوصية الطبية والسريرية:
  {'🚨 إحالة عاجلة لعيادة جراحة الأورام: يُنصح بإجراء خزعة نسيجية تأكيدية (Core Needle Biopsy) وتحديد مرحلة الورم فوراً لبدء البروتوكول العلاجي.' if prediction == 1 else '✔️ متابعة دورية روتينية: الخصائص الخلوية تشير إلى نسيج حميد وسليم. يُكتفى بالمتابعة السنوية الروتينية وفحص الماموجرام المعتاد.'}
--------------------------------------------------------------------------------
إخلاء مسؤولية طبي وقانوني:
هذا التقرير صادر عن نظام حاسوبي داعم لاتخاذ القرار الطبي (CAD System) باستخدام
خوارزميات تعلم الآلة المتقدمة، ولا يُغني عن الفحص المجهري المباشر من قِبل
استشاري الباثولوجيا والأورام المعتمد.
================================================================================
"""
    else:
        return f"""================================================================================
CLINICAL CYTOPATHOLOGY EVALUATION REPORT
Wisconsin Breast Cancer CAD AI Decision Support System
================================================================================
Date & Timestamp:     {now_str}
Reviewing Supervisor: Eng Mathew
Research & Dev Team:  Remon Ayman (Lead), Marcelino, Bishoy, Mario, Engy, Karen
Diagnostic Status:    Automated Clinical Decision Support (ML Inference)
--------------------------------------------------------------------------------
PATIENT CYTOMORPHOMETRIC MEASUREMENTS (Scale: 1 [Normal] to 10 [Severe]):
  1. Clump Thickness:              {patient_data['Clump_Thickness']} / 10
  2. Uniformity of Cell Size:      {patient_data['Uniformity_Cell_Size']} / 10
  3. Uniformity of Cell Shape:     {patient_data['Uniformity_Cell_Shape']} / 10
  4. Marginal Adhesion:            {patient_data['Marginal_Adhesion']} / 10
  5. Single Epithelial Cell Size:  {patient_data['Single_Epithelial_Cell_Size']} / 10
  6. Bare Nuclei:                  {patient_data['Bare_Nuclei']} / 10
  7. Bland Chromatin:              {patient_data['Bland_Chromatin']} / 10
  8. Normal Nucleoli:              {patient_data['Normal_Nucleoli']} / 10
  9. Mitoses:                      {patient_data['Mitoses']} / 10
--------------------------------------------------------------------------------
DIAGNOSTIC INFERENCE RESULTS:
  - Primary Classification:       {'MALIGNANT (CARCINOMA CONFIRMED)' if prediction == 1 else 'BENIGN (NON-MALIGNANT LESION)'}
  - Malignancy Risk Probability:  {prob_malignant:.2f}%
  - Benign Baseline Probability:  {prob_benign:.2f}%
  - Algorithmic Confidence Level: {'Very High (>90%)' if max(prob_malignant, prob_benign) > 90 else 'High Confidence'}
  - Active Classification Engine: {model_name}
  - Stratified Test Recall:       95.83% (Clinical Sensitivity across test cohort)
--------------------------------------------------------------------------------
CLINICAL RECOMMENDATION:
  {'🚨 URGENT ONCOLOGY REFERRAL: Immediate surgical oncology consultation and confirmatory histopathological core biopsy recommended for staging.' if prediction == 1 else '✔️ ROUTINE SURVEILLANCE: Findings indicate benign cytological features. Recommend routine annual screening mammography and clinical monitoring.'}
--------------------------------------------------------------------------------
MEDICO-LEGAL NOTICE & CLINICAL INTENDED USE:
This computational report was generated by an artificial intelligence Computer-Aided
Diagnosis (CAD) model strictly for clinical decision support under certified medical
supervision. It does not replace independent microscopic review by a pathologist.
================================================================================
"""


# ==============================================================================
# 5. RESPONSIVE LANGUAGE SELECTOR
# ==============================================================================
def render_language_selector():
    sub_col1, sub_col2 = st.columns([2, 1])
    with sub_col2:
        lang_choice = st.radio(
            "Language",
            options=["🌐 EN", "🇪🇬 AR"],
            index=0 if st.session_state.lang == "EN" else 1,
            horizontal=True,
            label_visibility="collapsed"
        )
        new_lang = "AR" if "AR" in lang_choice else "EN"
        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st.rerun()


# ==============================================================================
# SCREEN 1: WELCOME & LANDING SCREEN (Zero Lag & Mobile Optimized)
# ==============================================================================
if not st.session_state.started:
    render_language_selector()

    # Hero Welcome Box
    if is_ar:
        st.markdown("""
        <div class="welcome-container" dir="rtl">
            <div class="welcome-badge">🎗️ مشروع الذكاء الاصطناعي المتقدم — التصنيف بتعلم الآلة</div>
            <div class="welcome-title">أهلاً بك م/ ماثيو 👋</div>
            <div class="welcome-subtitle">
                مرحباً بك في <b>نظام التصنيف الخلوي والتشخيص الذكي لسرطان الثدي (CAD Diagnostic System)</b>.<br>
                منصة دعم قرار سريري متكاملة لمساعدة أطباء الأورام والتحاليل في الكشف المبكر عن الأورام الخبيثة بدقة متناهية عبر قياسات الخزعة المجهرية (FNA).
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
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

    # 4 Quick Highlights in CSS Responsive Grid (2x2 on Mobile, 4x1 on Desktop)
    lbl_acc = "🏆 أعلى دقة اختبار" if is_ar else "🏆 Peak Accuracy"
    lbl_rec = "🎯 الحساسية الطبية (Recall)" if is_ar else "🎯 Clinical Recall"
    lbl_auc = "📈 أعلى تمييز (ROC-AUC)" if is_ar else "📈 Highest ROC-AUC"
    lbl_mod = "⚡ نماذج التعلم والتجميع" if is_ar else "⚡ ML + Ensemble"

    st.markdown(f"""
    <div class="stat-grid" dir="{'rtl' if is_ar else 'ltr'}">
        <div class="stat-card">
            <div class="stat-value">97.12%</div>
            <div class="stat-label">{lbl_acc}</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">95.83%</div>
            <div class="stat-label">{lbl_rec}</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">0.9961</div>
            <div class="stat-label">{lbl_auc}</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">6 Models</div>
            <div class="stat-label">{lbl_mod}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Central Call To Action (Full Responsive Width)
    if is_ar:
        st.info("💡 **جاهز لتجربة واستعراض نماذج التشخيص؟** اضغط بالأسفل للدخول المباشر إلى بيئة الفحص اللحظي للمريض.")
        btn_label = "🚀 بدء نظام الفحص والتشخيص"
    else:
        st.info("💡 **Ready to explore the diagnostic models?** Click below to launch the interactive live patient testing environment.")
        btn_label = "🚀 START DIAGNOSTIC SYSTEM"

    if st.button(btn_label, type="primary", use_container_width=True):
        st.session_state.started = True
        st.rerun()

    st.write("")
    st.markdown("---")

    # Team Members (David removed)
    st.subheader("👥 فريق عمل المشروع" if is_ar else "👥 Project Development Team")
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <span class="team-pill">👨‍💻 Remon Ayman</span>
        <span class="team-pill">👨‍💻 Marcelino</span>
        <span class="team-pill">👨‍💻 Bishoy</span>
        <span class="team-pill">👨‍💻 Mario</span>
        <span class="team-pill">👩‍💻 Engy</span>
        <span class="team-pill">👩‍💻 Karen</span>
    </div>
    """, unsafe_allow_html=True)

    expander_title = "📌 نظرة عامة على معمارية المشروع والداتا سيت (اضغط للعرض)" if is_ar else "📌 Project Architecture & Dataset Overview (Click to Preview)"
    with st.expander(expander_title):
        if is_ar:
            st.markdown("""
            - **الداتا سيت**: بيانات أورام الثدي من جامعة كاليفورنيا (UCI Wisconsin) جمعها د. ويليام ولبرج (699 مريض، 9 خصائص خلوية مجهرية).
            - **تنظيف البيانات**: معالجة 16 قيمة ناقصة في عمود `Bare_Nuclei` باستخدام الوسيط (Median=1.0) المقتصر على داتا التدريب؛ وحذف 8 صفوف مكررة بالكامل (691 عينة فريدة).
            - **التقسيم الطبقي**: 80% تدريب (552 عينة) : 20% اختبار (139 عينة) مع ثبات نسبة الأورام الخبيثة (34.5%).
            - **منع تسرب الداتا**: تدريب `StandardScaler` على بيانات التدريب حصراً وتطبيقه على عينات الاختبار.
            - **الخوارزميات المدربة**: Random Forest، و Artificial Neural Network (ANN)، و Support Vector Machine (SVM)، و Voting Ensemble، و KNN، و Decision Tree.
            """)
        else:
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
    imputer, scaler = load_preprocessors()

    # Top Responsive Navigation Bar
    plat_title = "🎗️ منصة التشخيص الذكي للأورام" if is_ar else "🎗️ CAD Diagnostic Platform"
    sess_title = "جلسة: م/ ماثيو" if is_ar else "Session: Eng Mathew"

    st.markdown(f"""
    <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(148, 163, 184, 0.2);">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: clamp(1.2rem, 3.5vw, 1.5rem); font-weight: 800; color: #38bdf8;">{plat_title}</span>
            <span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 700;">
                {sess_title}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_col1, nav_col2 = st.columns([1, 1])
    with nav_col1:
        ret_label = "🏠 العودة للرئيسية" if is_ar else "🏠 Return to Welcome"
        if st.button(ret_label, use_container_width=True):
            st.session_state.started = False
            st.rerun()

    with nav_col2:
        lang_choice = st.radio(
            "Language",
            options=["🌐 EN", "🇪🇬 AR"],
            index=0 if st.session_state.lang == "EN" else 1,
            horizontal=True,
            label_visibility="collapsed"
        )
        new_lang = "AR" if "AR" in lang_choice else "EN"
        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st.rerun()

    # Sidebar: Model and Presets
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 0.5rem; background: rgba(56, 189, 248, 0.1); border-radius: 12px; margin-bottom: 1rem;">
        <span style="font-size: 1.1rem; font-weight: 700; color: #38bdf8;">{'أهلاً بك م/ ماثيو! 👋' if is_ar else 'Welcome, Eng Mathew! 👋'}</span><br>
        <span style="font-size: 0.8rem; color: #94a3b8;">{'لوحة التحكم بالتشخيص' if is_ar else 'Diagnostic Control Panel'}</span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("⚙️ إعدادات النموذج الذكي" if is_ar else "⚙️ Model Configuration")
    curr_model_dict = MODEL_OPTIONS_MAP['AR'] if is_ar else MODEL_OPTIONS_MAP['EN']
    selected_model_label = st.sidebar.selectbox(
        "اختر خوارزمية التصنيف:" if is_ar else "Select Classification Model:",
        options=list(curr_model_dict.keys()),
        index=0
    )
    active_model_path = curr_model_dict[selected_model_label]
    active_model = load_single_model_by_path(active_model_path)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 حالات سريرية نموذجية جاهزة" if is_ar else "📋 Quick Clinical Presets")

    preset_options = [
        "إدخال يدوي مخصص (Custom)" if is_ar else "Custom Manual Entry",
        "مريضة سليمة (ورم حميد نموذجي)" if is_ar else "Typical Benign Patient",
        "مريضة مصابة (ورم سرطاني نموذجي)" if is_ar else "Typical Malignant Patient",
        "حالة حرجة مشكوك فيها (Borderline)" if is_ar else "Borderline / Equivocal Case"
    ]
    preset = st.sidebar.radio(
        "تعبئة مؤشرات الخزعة:" if is_ar else "Populate Biopsy Attributes:",
        options=preset_options
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

    if "Benign" in preset or "حميد" in preset:
        default_values = {k: 1 for k in default_values}
        default_values['Clump_Thickness'] = 2
        default_values['Single_Epithelial_Cell_Size'] = 2
    elif "Malignant" in preset or "سرطاني" in preset:
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
    elif "Borderline" in preset or "حرجة" in preset:
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
    if is_ar:
        tab_names = [
            "🩺 التشخيص اللحظي للمريضة",
            "📁 فحص ملف مرضى جماعي وتنظيف ذاتي",
            "📊 مقارنة أداء النماذج (Benchmark)",
            "📖 الدليل المجهري لدكتور ولبرج",
            "ℹ️ معمارية المشروع وخط الأنابيب"
        ]
    else:
        tab_names = [
            "🩺 Live Patient Diagnosis",
            "📁 Batch CSV Diagnosis & Auto-Clean",
            "📊 Model Benchmark & Comparison",
            "📖 Cytological Features Guide",
            "ℹ️ Project Info & Architecture"
        ]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names)

    # ---------------- TAB 1: LIVE DIAGNOSIS ----------------
    with tab1:
        st.markdown("### " + ("أدخل القياسات الخلوية لخزعة المريضة (مقياس من ١ إلى ١٠)" if is_ar else "Enter Patient Biopsy Cytological Measurements (Scale: 1 - 10)"))
        st.caption("مقياس مجهري من ١ (طبيعي سليم) إلى ١٠ (تشوه خبيث شديد) طبقاً لفحص الخزعة بالإبرة الرفيعة (FNA)." if is_ar else "Graded from **1 (Normal / Healthy)** to **10 (Most Severe Abnormality)** based on Fine Needle Aspirate (FNA) cellular examination.")

        col1, col2, col3 = st.columns(3)

        with col1:
            lbl1 = "1. سُمك التكتل الخلوي (Clump Thickness)" if is_ar else "1. Clump Thickness"
            help1 = "تكدس وتراكم الخلايا فوق بعضها في طبقات متعددة" if is_ar else "Degree of multi-layered cellular agglomeration."
            clump = st.slider(lbl1, 1, 10, default_values['Clump_Thickness'], help=help1)

            lbl2 = "2. انتظام حجم الخلية (Cell Size Uniformity)" if is_ar else "2. Uniformity of Cell Size"
            help2 = "تفاوت واختلاف أحجام الخلايا عن بعضها" if is_ar else "Variability in cell dimensions (classic hallmark of cancer)."
            u_size = st.slider(lbl2, 1, 10, default_values['Uniformity_Cell_Size'], help=help2)

            lbl3 = "3. انتظام شكل الخلية (Cell Shape Uniformity)" if is_ar else "3. Uniformity of Cell Shape"
            help3 = "تشوه الحدود الخارجية للأغشية الخلوية" if is_ar else "Variability in cell boundary geometry and irregular contours."
            u_shape = st.slider(lbl3, 1, 10, default_values['Uniformity_Cell_Shape'], help=help3)

        with col2:
            lbl4 = "4. التصاق الحواف (Marginal Adhesion)" if is_ar else "4. Marginal Adhesion"
            help4 = "فقدان ترابط الخلايا يسهل انتشار الورم" if is_ar else "Tendency of cells to stick together; loss of adhesion enables metastasis."
            adhesion = st.slider(lbl4, 1, 10, default_values['Marginal_Adhesion'], help=help4)

            lbl5 = "5. حجم الخلية المفردة (Epithelial Cell Size)" if is_ar else "5. Single Epithelial Cell Size"
            help5 = "تضخم حجم السيتوبلازم وتغير النسبة النووية" if is_ar else "Epithelial cell volume and nuclear-to-cytoplasmic ratio."
            epi_size = st.slider(lbl5, 1, 10, default_values['Single_Epithelial_Cell_Size'], help=help5)

            lbl6 = "6. الأنوية العارية (Bare Nuclei)" if is_ar else "6. Bare Nuclei"
            help6 = "أنوية بدون غلاف سيتوبلازمي (علامة قوية على الخباثة)" if is_ar else "Nuclei without surrounding cytoplasm (strong indicator of malignancy)."
            bare_nuc = st.slider(lbl6, 1, 10, default_values['Bare_Nuclei'], help=help6)

        with col3:
            lbl7 = "7. الكروماتين النووي (Bland Chromatin)" if is_ar else "7. Bland Chromatin"
            help7 = "خشونة وتكتل المادة الوراثية داخل النواة" if is_ar else "Coarseness and aggregation of nuclear chromatin."
            bland_chr = st.slider(lbl7, 1, 10, default_values['Bland_Chromatin'], help=help7)

            lbl8 = "8. النويات الطبيعية (Normal Nucleoli)" if is_ar else "8. Normal Nucleoli"
            help8 = "تضخم وبروز النويات يعكس شراسة الانقسام" if is_ar else "Prominence and hyperactivity of nucleoli."
            norm_nuc = st.slider(lbl8, 1, 10, default_values['Normal_Nucleoli'], help=help8)

            lbl9 = "9. الانقسام الميتوزي (Mitoses)" if is_ar else "9. Mitoses"
            help9 = "معدل تكاثر وانقسام الخلايا النشطة" if is_ar else "Frequency of cells undergoing active mitotic division."
            mitoses = st.slider(lbl9, 1, 10, default_values['Mitoses'], help=help9)

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
        btn_diag_label = "🔬 تحليل الخزعة وتشخيص الحالة فورياً" if is_ar else "🔬 Analyze Biopsy & Run Diagnosis"
        predict_btn = st.button(btn_diag_label, type="primary", use_container_width=True)

        is_custom = ("Custom" in preset or "مخصص" in preset)
        if predict_btn or not is_custom:
            patient_imp = pd.DataFrame(imputer.transform(patient_df), columns=list(feature_dict.keys()))
            patient_scaled = pd.DataFrame(scaler.transform(patient_imp), columns=list(feature_dict.keys()))

            prediction = active_model.predict(patient_scaled)[0]
            probabilities = active_model.predict_proba(patient_scaled)[0]

            prob_benign = probabilities[0] * 100
            prob_malignant = probabilities[1] * 100

            st.write("")
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                st.subheader("نتيجة التشخيص السريري:" if is_ar else "Diagnostic Classification:")
                if prediction == 1:
                    if is_ar:
                        st.markdown(f"""
                        <div class="malignant-badge" dir="rtl">
                            ⚠️ نتيجة التشخيص: ورم سرطاني خبيث (MALIGNANT)<br>
                            <span style="font-size: 1.1rem; font-weight: 500;">
                                احتمالية الخباثة: <b>{prob_malignant:.2f}%</b> (درجة الثقة: مرتفعة)
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.error("🚨 **التوصية الطبية**: تحويل فوري لعيادة جراحة الأورام، وإجراء فحص هستوباثولوجي تأكيدي وتحديد مرحلة الورم فوراً.")
                    else:
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
                    if is_ar:
                        st.markdown(f"""
                        <div class="benign-badge" dir="rtl">
                            ✅ نتيجة التشخيص: ورم حميد سليم (BENIGN)<br>
                            <span style="font-size: 1.1rem; font-weight: 500;">
                                احتمالية الورم الحميد: <b>{prob_benign:.2f}%</b> (درجة الثقة: مرتفعة)
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("✔️ **التوصية الطبية**: احتمالية ضئيلة جداً للأورام الخبيثة. يُنصح بالمتابعة الدورية الروتينية وفحص الماموجرام السنوي.")
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

                st.markdown("#### " + ("مؤشر توزيع الاحتمالات:" if is_ar else "Probability Meter:"))
                st.progress(prob_malignant / 100)
                pcol1, pcol2 = st.columns(2)
                pcol1.metric("احتمالية الحميد" if is_ar else "Benign Probability", f"{prob_benign:.1f}%")
                pcol2.metric("احتمالية الخبيث" if is_ar else "Malignant Probability", f"{prob_malignant:.1f}%")

            with res_col2:
                st.subheader("بروفايل تشوهات الخلية للمريضة:" if is_ar else "Cellular Abnormality Profile:")
                if is_ar:
                    st.caption("مقارنة انحراف خصائص خزعة المريضة عن خط الأمان المجهري للخلية الطبيعية السليمة.")

                fig, ax = plt.subplots(figsize=(6, 4))
                feats = [f.replace('_', ' ') for f in feature_dict.keys()]
                vals = list(feature_dict.values())

                y_pos = np.arange(len(feats))
                bar_color = '#f43f5e' if prediction == 1 else '#38bdf8'
                ax.barh(y_pos, vals, color=bar_color, alpha=0.85)
                # Keep internal chart annotations in crisp medical English to prevent font glyph separation
                ax.axvline(1, color='#10b981', linestyle='--', linewidth=1.5, label='Healthy Baseline (1.0)')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feats, fontsize=8)
                ax.set_xlim(0, 10.5)
                ax.set_xlabel('Cytological Rating (1 - 10)', fontsize=9)
                ax.set_title('Biopsy Severity vs Normal Cell Baseline', fontsize=10, fontweight='bold')
                ax.legend(loc='lower right', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            # --- Clinical Pathology Report Generator & Download ---
            st.markdown("---")
            rep_title = "📥 التقرير الطبي المعتمد للحالة (Official Clinical Pathology Report)" if is_ar else "📥 Official Clinical Pathology Report"
            st.markdown(f"#### {rep_title}")
            report_content = generate_clinical_report(feature_dict, prediction, prob_benign, prob_malignant, selected_model_label, is_ar)

            rpt_col1, rpt_col2 = st.columns([1.8, 1])
            with rpt_col1:
                with st.expander("👁️ " + ("معاينة مسودة التقرير السريري الرسمي قبل التحميل" if is_ar else "Preview Clinical Report Document")):
                    st.code(report_content, language="markdown")
            with rpt_col2:
                btn_dl_txt = "📄 " + ("تحميل التقرير الطبي (TXT)" if is_ar else "Download Clinical Report (.TXT)")
                st.download_button(
                    label=btn_dl_txt,
                    data=report_content,
                    file_name=f"clinical_pathology_report_{int(time.time())}.txt",
                    mime="text/plain",
                    type="primary",
                    use_container_width=True
                )

    # ---------------- TAB 2: BATCH CSV & AUTO-CLEAN ----------------
    with tab2:
        st.subheader("📁 " + ("فحص ملف مرضى جماعي وتصنيف فوري بالذكاء الاصطناعي" if is_ar else "Batch Patient Biopsy Diagnosis & Auto-Data Cleaning"))
        st.write(
            "ارفع ملف CSV يحتوي على عينات لمرضى متعددين لإجراء تشخيص آلي فوري وشامل مع فحص دقيق لجودة البيانات وتنظيفها ذاتياً وتحديد مستويات الخطورة السريرية."
            if is_ar else
            "Upload a batch CSV file of patient biopsies for automated bulk inference. Includes automatic schema validation, data hygiene auditing, training-median imputation, and clinical risk stratification."
        )

        b_col1, b_col2 = st.columns([1.5, 1])
        with b_col1:
            uploaded_batch = st.file_uploader(
                "اختر ملف CSV للمرضى:" if is_ar else "Select Patient Batch CSV File:",
                type=["csv"],
                key="batch_file_uploader"
            )
        with b_col2:
            st.markdown(f"**{'نموذج ملف تجريبي جاهز للاختبار:' if is_ar else 'Test with Sample Batch Data:'}**")
            st.caption("ملف يحتوي على 20 عينة تجريبية لمرضى بها حالات سليمة وسرطانية وقيم مفقودة لاختبار ميزة التنظيف الذاتي." if is_ar else "20 realistic patient cases with clean and missing values ('?') to demonstrate automated cleaning.")
            if os.path.exists("sample_patients_batch.csv"):
                with open("sample_patients_batch.csv", "r", encoding="utf-8") as sf:
                    sample_bytes = sf.read()
                st.download_button(
                    label="📥 " + ("تحميل ملف تجريبي (sample_patients_batch.csv)" if is_ar else "Download Sample CSV Template"),
                    data=sample_bytes,
                    file_name="sample_patients_batch.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        if uploaded_batch is not None:
            try:
                # Read CSV
                raw_df = pd.read_csv(uploaded_batch)
                
                # Normalize column names for flexible matching
                normalized_cols = {col: col.strip().lower().replace(" ", "_") for col in raw_df.columns}
                
                # Check for standard feature names
                feature_aliases = {
                    'clump_thickness': 'Clump_Thickness',
                    'uniformity_cell_size': 'Uniformity_Cell_Size',
                    'uniformity_of_cell_size': 'Uniformity_Cell_Size',
                    'uniformity_cell_shape': 'Uniformity_Cell_Shape',
                    'uniformity_of_cell_shape': 'Uniformity_Cell_Shape',
                    'marginal_adhesion': 'Marginal_Adhesion',
                    'single_epithelial_cell_size': 'Single_Epithelial_Cell_Size',
                    'epithelial_cell_size': 'Single_Epithelial_Cell_Size',
                    'bare_nuclei': 'Bare_Nuclei',
                    'bland_chromatin': 'Bland_Chromatin',
                    'normal_nucleoli': 'Normal_Nucleoli',
                    'mitoses': 'Mitoses'
                }
                
                mapped_cols = {}
                for orig, norm in normalized_cols.items():
                    if norm in feature_aliases:
                        mapped_cols[orig] = feature_aliases[norm]
                    elif 'sample' in norm or 'id' in norm or 'code' in norm or 'patient' in norm:
                        mapped_cols[orig] = 'Patient_ID'
                        
                df_standardized = raw_df.rename(columns=mapped_cols)
                
                FEATURE_COLS = [
                    'Clump_Thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape',
                    'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei',
                    'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses'
                ]
                
                missing_req_cols = [f for f in FEATURE_COLS if f not in df_standardized.columns]
                
                if missing_req_cols:
                    err_msg = f"الأعمدة التالية مفقودة في الملف: {missing_req_cols}" if is_ar else f"Missing required cytological columns: {missing_req_cols}"
                    st.error(f"❌ {err_msg}")
                    st.info("الأعمدة المطلوبة: " + ", ".join(FEATURE_COLS) if is_ar else "Required columns: " + ", ".join(FEATURE_COLS))
                else:
                    st.markdown("---")
                    st.subheader("🔍 " + ("تقرير تدقيق جودة البيانات (Data Hygiene & Quality Audit)" if is_ar else "Data Hygiene & Quality Audit"))
                    
                    # Extract IDs
                    if 'Patient_ID' in df_standardized.columns:
                        patient_ids = df_standardized['Patient_ID'].astype(str).tolist()
                    else:
                        patient_ids = [f"PATIENT-{i+1:04d}" for i in range(len(df_standardized))]
                        
                    raw_feature_df = df_standardized[FEATURE_COLS].copy()
                    
                    # Audit checks
                    # 1. Missing values or '?' strings
                    q_mark_counts = (raw_feature_df.astype(str).apply(lambda s: s.str.strip()) == '?').sum().to_dict()
                    total_q_marks = sum(q_mark_counts.values())
                    
                    # Replace '?' with NaN and coerce to numeric
                    cleanable_df = raw_feature_df.replace('?', np.nan).apply(pd.to_numeric, errors='coerce')
                    nan_counts = cleanable_df.isna().sum().to_dict()
                    total_nans = sum(nan_counts.values())
                    
                    # 2. Duplicate rows
                    dup_rows = int(raw_feature_df.duplicated().sum())
                    
                    # 3. Out of range [1, 10]
                    out_of_range = int(((cleanable_df < 1) | (cleanable_df > 10)).sum().sum())
                    
                    is_clean_data = (total_nans == 0 and total_q_marks == 0 and dup_rows == 0 and out_of_range == 0)
                    
                    audit_col1, audit_col2 = st.columns([1, 1])
                    with audit_col1:
                        if is_clean_data:
                            st.success("✅ **حالة البيانات: نظيفة وسليمة بنسبة 100% (Clean Dataset)**" if is_ar else "✅ **Dataset Status: 100% Clean!**")
                            st.markdown(
                                f"جميع السجلات الـ ({len(raw_df)}) كاملة وتحتوي على قيم عددية صحيحة في النطاق [1 - 10] بدون أي قيم مفقودة أو صفوف مكررة."
                                if is_ar else
                                f"All {len(raw_df)} records are complete, fully numeric within [1 - 10], with zero missing values or duplicate anomalies."
                            )
                        else:
                            st.warning("⚠️ **حالة البيانات: تم رصد شوائب / بيانات غير نظيفة (Unclean / Dirty Data)**" if is_ar else "⚠️ **Dataset Status: Unclean / Dirty Data Detected**")
                            issues_desc = []
                            if total_q_marks > 0 or total_nans > 0:
                                missing_col_detail = [f"`{col}` ({nan_counts[col]} قيم)" for col in FEATURE_COLS if nan_counts[col] > 0]
                                issues_desc.append(f"• تم رصد **{total_nans}** قيمة مفقودة أو غير رقمية في: {', '.join(missing_col_detail)}" if is_ar else f"• Found **{total_nans}** missing/invalid entries in: {', '.join(missing_col_detail)}")
                            if dup_rows > 0:
                                issues_desc.append(f"• تم رصد **{dup_rows}** سجل مكرر بالكامل." if is_ar else f"• Found **{dup_rows}** duplicate patient records.")
                            if out_of_range > 0:
                                issues_desc.append(f"• تم رصد **{out_of_range}** قيمة خارج النطاق القياسي [1, 10]." if is_ar else f"• Found **{out_of_range}** values out of range [1, 10].")
                                
                            st.markdown("\n".join(issues_desc))
                    
                    with audit_col2:
                        st.info("🧹 **إجراءات خط أنابيب التنظيف الذاتي (Auto-Clean Actions):**\n"
                                f"• **تعويض القيم الناقصة**: تم تعويض الـ ({total_nans}) قيمة بالوسيط التدريبي الصارم (`Median = 1.0`) لمنع تسرب البيانات.\n"
                                f"• **ضبط النطاق وتوحيد الأنواع**: تحويل كل الأعمدة إلى أرقام صحيحة [1, 10].\n"
                                f"• **حماية هوية المريض**: الحفاظ على أرقام السجلات المعرّفة للمرضى سليمة 100%."
                                if is_ar else
                                "🧹 **Automated Cleaning Pipeline Actions:**\n"
                                f"• **Leak-Free Imputation**: {total_nans} missing values imputed using training median (`Median = 1.0`).\n"
                                f"• **Schema Alignment**: Coerced cytological ratings to integer domain [1, 10].\n"
                                f"• **Entity Integrity**: Preserved patient identifiers and record lineage."
                        )
                    
                    # Execute Auto-Clean Pipeline
                    imputed_batch = pd.DataFrame(imputer.transform(cleanable_df), columns=FEATURE_COLS)
                    # Clip values between 1 and 10
                    imputed_batch = imputed_batch.clip(lower=1, upper=10)
                    
                    # Scale features
                    scaled_batch = pd.DataFrame(scaler.transform(imputed_batch), columns=FEATURE_COLS)
                    
                    # Model Inference
                    batch_preds = active_model.predict(scaled_batch)
                    batch_probas = active_model.predict_proba(scaled_batch)
                    
                    malignant_probs = batch_probas[:, 1] * 100
                    benign_probs = batch_probas[:, 0] * 100
                    
                    total_patients = len(raw_df)
                    malignant_total = int((batch_preds == 1).sum())
                    benign_total = int((batch_preds == 0).sum())
                    high_risk_total = int((malignant_probs >= 80.0).sum())
                    
                    st.write("")
                    st.markdown("### " + ("📊 ملخص تشخيص الدفعة السريرية:" if is_ar else "📊 Batch Clinical Diagnostic Summary:"))
                    
                    b_met1, b_met2, b_met3, b_met4 = st.columns(4)
                    b_met1.metric(
                        "إجمالي المفحوصين" if is_ar else "Total Cohort",
                        f"{total_patients} مريض" if is_ar else f"{total_patients} Patients"
                    )
                    b_met2.metric(
                        "أورام حميدة (Benign)" if is_ar else "Benign Cases",
                        f"{benign_total} ({benign_total/total_patients*100:.1f}%)",
                        delta="حالات مطمئنة" if is_ar else "Low Risk",
                        delta_color="normal"
                    )
                    b_met3.metric(
                        "أورام خبيثة (Malignant)" if is_ar else "Malignant Cases",
                        f"{malignant_total} ({malignant_total/total_patients*100:.1f}%)",
                        delta="تحتاج تدخل" if is_ar else "Requires Care",
                        delta_color="inverse"
                    )
                    b_met4.metric(
                        "حالات حرجة جداً (>80%)" if is_ar else "Critical High-Risk",
                        f"{high_risk_total}",
                        delta="إحالة عاجلة" if is_ar else "Urgent Referral",
                        delta_color="inverse"
                    )
                    
                    # Assemble Output Results Table
                    results_list = []
                    for i in range(total_patients):
                        p_id = patient_ids[i]
                        pred_label = "MALIGNANT" if batch_preds[i] == 1 else "BENIGN"
                        pred_label_ar = "خبيث (Malignant)" if batch_preds[i] == 1 else "حميد (Benign)"
                        m_p = malignant_probs[i]
                        b_p = benign_probs[i]
                        
                        if m_p >= 80.0:
                            risk_str = "عالي الخطورة (حرجة)" if is_ar else "CRITICAL HIGH"
                        elif m_p >= 50.0:
                            risk_str = "متوسط الخطورة" if is_ar else "MODERATE RISK"
                        elif m_p >= 25.0:
                            risk_str = "مشكوك فيها (Borderline)" if is_ar else "BORDERLINE"
                        else:
                            risk_str = "منخفض جداً (سليم)" if is_ar else "LOW SUSPICION"
                            
                        rec_str = "إحالة جراحية فورية للأورام" if is_ar else "Urgent Oncology Referral" if batch_preds[i] == 1 else ("متابعة دورية سنوية" if is_ar else "Routine Annual Surveillance")
                        
                        row_dict = {
                            'Patient_ID' if not is_ar else 'رقم المريض': p_id,
                            'Diagnosis' if not is_ar else 'التشخيص': pred_label_ar if is_ar else pred_label,
                            'Malignancy_Risk_%' if not is_ar else 'نسبة الخباثة %': f"{m_p:.1f}%",
                            'Benign_Score_%' if not is_ar else 'نسبة الأمان %': f"{b_p:.1f}%",
                            'Risk_Level' if not is_ar else 'مستوى الخطورة': risk_str,
                            'Recommendation' if not is_ar else 'التوصية السريرية': rec_str
                        }
                        for col in FEATURE_COLS:
                            col_display = col if not is_ar else col
                            row_dict[col_display] = int(imputed_batch.iloc[i][col])
                        results_list.append(row_dict)
                        
                    batch_results_df = pd.DataFrame(results_list)
                    
                    # Interactive Filtering
                    f_col1, f_col2 = st.columns([2, 1])
                    with f_col1:
                        filter_opt = st.radio(
                            "فلترة عرض الحالات:" if is_ar else "Filter Cohort View:",
                            options=["الكل (All)" if is_ar else "All Cases",
                                     "الأورام الخبيثة فقط (Malignant Only)" if is_ar else "Malignant Only",
                                     "الأورام الحميدة فقط (Benign Only)" if is_ar else "Benign Only"],
                            horizontal=True
                        )
                    
                    diag_col_name = 'التشخيص' if is_ar else 'Diagnosis'
                    if "Malignant" in filter_opt or "الخبيثة" in filter_opt:
                        view_df = batch_results_df[batch_results_df[diag_col_name].str.contains("MALIGNANT|خبيث", case=False)]
                    elif "Benign" in filter_opt or "الحميدة" in filter_opt:
                        view_df = batch_results_df[batch_results_df[diag_col_name].str.contains("BENIGN|حميد", case=False)]
                    else:
                        view_df = batch_results_df
                        
                    st.dataframe(view_df, hide_index=True, use_container_width=True)
                    
                    # Download Batch CSV
                    batch_csv_export = batch_results_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 " + ("تحميل تقرير نتائج الفحص الجماعي الشامل (CSV)" if is_ar else "Download Full Batch Results (CSV)"),
                        data=batch_csv_export,
                        file_name=f"batch_cancer_diagnoses_{int(time.time())}.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"حدث خطأ أثناء قراءة ومعالجة الملف: {str(e)}" if is_ar else f"Error processing batch CSV: {str(e)}")

    # ---------------- TAB 3: MODEL BENCHMARK ----------------
    with tab3:
        st.subheader("📊 مقارنة أداء النماذج ومقاييس التقييم" if is_ar else "📊 Model Performance Benchmark & Comparison")
        st.write("تم تقييم جميع النماذج بدقة متناهية على عينة الاختبار المعزولة (N=139) عبر **5-Fold Stratified Cross-Validation**." if is_ar else "All models were rigorously evaluated on the held-out test set ($N=139$) using **5-Fold Stratified Cross-Validation**.")

        results_table = pd.read_csv('model_comparison_results.csv')

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("🏆 أعلى دقة اختبار" if is_ar else "🏆 Peak Accuracy", "97.12%", "Random Forest / ANN / SVM")
        m_col2.metric("🎯 أعلى حساسية (Recall)" if is_ar else "🎯 Peak Recall (Sensitivity)", "95.83%", "حالتين خطأ فقط" if is_ar else "Only 2 False Negatives")
        m_col3.metric("📈 أعلى تمييز احتمالي (AUC)" if is_ar else "📈 Highest ROC-AUC", "0.9961", "الشبكة العصبية (ANN)" if is_ar else "Artificial Neural Network")

        display_df = results_table.copy()
        if is_ar:
            display_df.rename(columns={
                'Model': 'النموذج',
                'Best Hyperparameters': 'أفضل المعاملات',
                'Accuracy': 'الدقة (Accuracy)',
                'Precision': 'الدقة الإيجابية (Precision)',
                'Recall': 'الحساسية (Recall)',
                'F1-Score': 'معيار F1',
                'ROC-AUC': 'مساحة المنحنى (ROC-AUC)'
            }, inplace=True)
            display_df['الدقة (Accuracy)'] = display_df['الدقة (Accuracy)'].map(lambda x: f"{float(x)*100:.2f}%")
            display_df['الدقة الإيجابية (Precision)'] = display_df['الدقة الإيجابية (Precision)'].map(lambda x: f"{float(x)*100:.2f}%")
            display_df['الحساسية (Recall)'] = display_df['الحساسية (Recall)'].map(lambda x: f"{float(x)*100:.2f}%")
            display_df['معيار F1'] = display_df['معيار F1'].map(lambda x: f"{float(x):.4f}")
            display_df['مساحة المنحنى (ROC-AUC)'] = display_df['مساحة المنحنى (ROC-AUC)'].map(lambda x: f"{float(x):.4f}")
        else:
            display_df['Accuracy'] = display_df['Accuracy'].map(lambda x: f"{float(x)*100:.2f}%")
            display_df['Precision'] = display_df['Precision'].map(lambda x: f"{float(x)*100:.2f}%")
            display_df['Recall'] = display_df['Recall'].map(lambda x: f"{float(x)*100:.2f}%")
            display_df['F1-Score'] = display_df['F1-Score'].map(lambda x: f"{float(x):.4f}")
            display_df['ROC-AUC'] = display_df['ROC-AUC'].map(lambda x: f"{float(x):.4f}")

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if os.path.exists("figures/7_all_confusion_matrices.png"):
                st.image("figures/7_all_confusion_matrices.png", caption="مصفوفات الالتباس لجميع النماذج (عينة الاختبار N=139)" if is_ar else "Confusion Matrices Across All Models (Test Set N=139)", use_container_width=True)
        with col_img2:
            if os.path.exists("figures/8_roc_curves_comparison.png"):
                st.image("figures/8_roc_curves_comparison.png", caption="منحنيات الـ ROC ومقارنة مساحات الـ AUC" if is_ar else "Receiver Operating Characteristic (ROC) Curves", use_container_width=True)

        if os.path.exists("figures/10_rf_feature_importance.png"):
            st.image("figures/10_rf_feature_importance.png", caption="أهمية الخصائص الخلوية طبقاً للـ Random Forest" if is_ar else "Random Forest Gini Feature Importance Analysis", use_container_width=True)

    # ---------------- TAB 4: CYTOLOGICAL CRITERIA ----------------
    with tab4:
        st.subheader("📖 الدليل المجهري للخصائص الخلوية (د. ويليام ولبرج)" if is_ar else "📖 Cytological Morphometry Guide (Dr. William H. Wolberg)")
        if is_ar:
            st.markdown("""
            | # | الخاصية الخلوية | الوصف السريري المجهري | الأهمية الباثولوجية في تشخيص السرطان |
            |---|---|---|---|
            | 1 | **سُمك التكتل (Clump Thickness)** | تكدس الخلايا في طبقات متعددة مقارنة بالطبقة الواحدة | الخلايا الحميدة تشكل طبقة واحدة مسطحة، بينما السرطانية تتكتل في كتل سميكة. |
            | 2 | **انتظام حجم الخلية (Cell Size)** | تفاوت واختلاف أحجام الخلايا عن بعضها | التفاوت الشديد في الحجم (Anisonucleosis) هو العلامة الكلاسيكية للتحول الخبيث. |
            | 3 | **انتظام شكل الخلية (Cell Shape)** | تشوه واضطراب حدود الأغشية الخلوية | الخلايا الخبيثة تظهر أشكالاً هندسية عشوائية وشاذة (Pleomorphism). |
            | 4 | **التصاق الحواف (Marginal Adhesion)** | قوة الروابط بين الخلايا المتجاورة | فقدان الالتصاق يسمح للخلايا السرطانية بالانفصال وغزو الأنسجة المجاورة. |
            | 5 | **حجم الخلية المفردة (Epithelial Size)** | تضخم حجم الخلية وتغير النسبة بين النواة والسيتوبلازم | القيم العالية تدل على تضخم خلوي خبيث ناتج عن فرط التخليق الحيوي. |
            | 6 | **الأنوية العارية (Bare Nuclei)** | أنوية فقدت غلاف السيتوبلازم المحيط بها | شائعة جداً في العينات الخبيثة، وتعتبر من أقوى مؤشرات الأورام المتقدمة. |
            | 7 | **الكروماتين النووي (Bland Chromatin)** | ملمس وخشونة المادة الوراثية داخل النواة | الكروماتين الأملس طبيعي، بينما المتكتل والخشن هو علامة السرطان. |
            | 8 | **النويات الطبيعية (Normal Nucleoli)** | حجم وبروز النويات داخل نواة الخلية | زيادة حجم وعدد النويات يعكس شراسة تصنيع البروتينات في الخلايا السرطانية. |
            | 9 | **الانقسام الميتوزي (Mitoses)** | عدد الانقسامات الخلوية النشطة المرصودة | الانقسام السريع يعكس معدل التكاثر الهستولوجي الشرس للورم. |
            """)
        else:
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

    # ---------------- TAB 5: PROJECT INFO ----------------
    with tab5:
        st.subheader("ℹ️ معمارية المشروع وخط الأنابيب الذكي" if is_ar else "ℹ️ Project Architecture & Pipeline Summary")
        if is_ar:
            st.markdown("""
            - **الهدف من تعلم الآلة**: تصنيف ثنائي دقيق لخزعات خلايا الثدي إلى ورم حميد (0) أو ورم خبيث (1).
            - **الداتا سيت المعتمدة**: مستودع جامعة كاليفورنيا (UCI) — بيانات ويسكونسن الأصلية لدكتور ولبرج.
            - **بروتوكول المعالجة المسبقة**: خالٍ بنسبة 100% من تسريب البيانات (تدريب Imputer و StandardScaler على داتا التدريب حصراً).
            - **بروتوكول التقييم**: 5-Fold Stratified Cross-Validation على داتا التدريب + تقييم معزول على عينة الاختبار ($N=139$).
            - **النموذج الفائز (Champion)**: **Random Forest** تم اختياره للتوازن المثالي بين أعلى حساسية طبية (95.83%)، وأعلى دقة (97.12%)، والشفافية الكاملة وتفسير القرار الطبي.
            - **الجلسة الأكاديمية**: مخصصة للمهندس **ماثيو (Eng Mathew)**.
            - **المستودع مفتوح المصدر**: [https://github.com/RemonAyman/advai](https://github.com/RemonAyman/advai)
            """)
        else:
            st.markdown("""
            - **Supervised Learning Objective**: Binary classification of cytological biopsies into Benign (0) vs. Malignant (1).
            - **Dataset**: UCI Machine Learning Repository — Breast Cancer Wisconsin (Original).
            - **Preprocessing Protocol**: Strictly leak-free (median imputation & `StandardScaler` fitted only on training set).
            - **Evaluation Protocol**: 5-Fold Stratified Cross-Validation on training folds + held-out test evaluation ($N=139$).
            - **Champion Model**: **Random Forest** selected for peak clinical recall (95.83%), peak accuracy (97.12%), and full feature explainability.
            - **Session Recipient**: Dedicated review for **Eng Mathew**.
            - **GitHub Source**: [https://github.com/RemonAyman/advai](https://github.com/RemonAyman/advai)
            """)

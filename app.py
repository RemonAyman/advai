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
if "lang" not in st.session_state:
    st.session_state.lang = "EN" # Default is English

# 3. Custom CSS Styling (Dark & Light Mode Compatible, Sleek Medical Tech Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Cairo:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', 'Cairo', sans-serif;
    }

    /* Language Switcher Floating Badge */
    .lang-container {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.5rem;
    }

    /* Welcome Hero Container */
    .welcome-container {
        background: linear-gradient(135deg, rgba(14, 116, 144, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid rgba(14, 116, 144, 0.3);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        margin-top: 0.5rem;
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
        max-width: 820px;
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

RAW_MODEL_FILES = {
    'Random Forest': 'models/random_forest.joblib',
    'SVM': 'models/support_vector_machine_svm.joblib',
    'ANN': 'models/artificial_neural_network_ann.joblib',
    'Voting': 'models/ensemble_voting.joblib',
    'KNN': 'models/knn.joblib',
    'Decision Tree': 'models/decision_tree.joblib'
}

@st.cache_resource
def load_preprocessors():
    imp = joblib.load('models/imputer.joblib')
    scl = joblib.load('models/scaler.joblib')
    return imp, scl

@st.cache_resource
def load_single_model_by_path(path):
    return joblib.load(path)


# ==============================================================================
# 5. LANGUAGE CONTROL COMPONENT (🌐 EN | AR)
# ==============================================================================
def render_language_selector():
    lang_col1, lang_col2 = st.columns([8.5, 1.5])
    with lang_col2:
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

is_ar = (st.session_state.lang == "AR")


# ==============================================================================
# SCREEN 1: WELCOME & LANDING SCREEN (Loads instantaneously in 0.01s)
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
                منصة دعم قرار سريري متكاملة مصممة لمساعدة أطباء الأورام والتحاليل في الكشف المبكر عن الأورام الخبيثة بدقة متناهية عبر القياسات المجهرية لخلايا الخزعة (FNA).
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

    # 4 Quick Highlights
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">97.12%</div>
            <div class="stat-label">{'🏆 أعلى دقة اختبار' if is_ar else '🏆 Peak Accuracy'}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">95.83%</div>
            <div class="stat-label">{'🎯 الحساسية الطبية (Recall)' if is_ar else '🎯 Clinical Recall'}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">0.9961</div>
            <div class="stat-label">{'📈 أعلى تمييز (ROC-AUC)' if is_ar else '📈 Highest ROC-AUC'}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">6 Models</div>
            <div class="stat-label">{'⚡ نماذج التعلم والتجميع' if is_ar else '⚡ ML + Ensemble'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Central Call To Action
    start_col1, start_col2, start_col3 = st.columns([1, 1.8, 1])
    with start_col2:
        if is_ar:
            st.info("💡 **جاهز لتجربة واستعراض نماذج التشخيص؟** اضغط بالأسفل للدخول المباشر إلى بيئة الفحص اللحظي.")
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

    # Top Bar: Personalized Navigation & Language Switcher
    top_col1, top_col2, top_col3 = st.columns([3, 1.2, 0.8])
    with top_col1:
        plat_title = "🎗️ منصة التشخيص الذكي للأورام" if is_ar else "🎗️ CAD Diagnostic Platform"
        sess_title = "جلسة الفحص: م/ ماثيو" if is_ar else "Session: Eng Mathew"
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 5px;">
            <span style="font-size: 1.5rem; font-weight: 800; color: #38bdf8;">{plat_title}</span>
            <span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 3px 10px; border-radius: 20px; font-size: 0.85rem; font-weight: 700;">
                {sess_title}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with top_col2:
        ret_label = "🏠 العودة للرئيسية" if is_ar else "🏠 Return to Welcome"
        if st.button(ret_label, use_container_width=True):
            st.session_state.started = False
            st.rerun()

    with top_col3:
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
            "📊 مقارنة أداء النماذج (Benchmark)",
            "📖 الدليل المجهري لدكتور ولبرج",
            "ℹ️ معمارية المشروع وخط الأنابيب"
        ]
    else:
        tab_names = [
            "🩺 Live Patient Diagnosis",
            "📊 Model Benchmark & Comparison",
            "📖 Cytological Features Guide",
            "ℹ️ Project Info & Architecture"
        ]

    tab1, tab2, tab3, tab4 = st.tabs(tab_names)

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
                fig, ax = plt.subplots(figsize=(6, 4))
                feats = [f.replace('_', ' ') for f in feature_dict.keys()]
                vals = list(feature_dict.values())

                y_pos = np.arange(len(feats))
                bar_color = '#f43f5e' if prediction == 1 else '#38bdf8'
                ax.barh(y_pos, vals, color=bar_color, alpha=0.85)
                base_lbl = 'الحد الطبيعي السليم (1.0)' if is_ar else 'Healthy Baseline (1.0)'
                ax.axvline(1, color='#10b981', linestyle='--', linewidth=1.5, label=base_lbl)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feats, fontsize=8)
                ax.set_xlim(0, 10.5)
                ax.set_xlabel('التقييم المجهري (1 - 10)' if is_ar else 'Cytological Rating (1 - 10)', fontsize=9)
                ax.set_title('مقارنة شدة الخصائص بالخلية الطبيعية السليمة' if is_ar else 'Feature Severity vs Normal Cell Baseline', fontsize=10, fontweight='bold')
                ax.legend(loc='lower right', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)

    # ---------------- TAB 2: MODEL BENCHMARK ----------------
    with tab2:
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
                st.image("figures/7_all_confusion_matrices.png", caption="مصفوفات الالتباس لجميع النماذج (عينة الاختبار N=139)" if is_ar else "Confusion Matrices Across All Models (Test Set N=139)")
        with col_img2:
            if os.path.exists("figures/8_roc_curves_comparison.png"):
                st.image("figures/8_roc_curves_comparison.png", caption="منحنيات الـ ROC ومقارنة مساحات الـ AUC" if is_ar else "Receiver Operating Characteristic (ROC) Curves")

        if os.path.exists("figures/10_rf_feature_importance.png"):
            st.image("figures/10_rf_feature_importance.png", caption="أهمية الخصائص الخلوية طبقاً للـ Random Forest" if is_ar else "Random Forest Gini Feature Importance Analysis")

    # ---------------- TAB 3: CYTOLOGICAL CRITERIA ----------------
    with tab3:
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

    # ---------------- TAB 4: PROJECT INFO ----------------
    with tab4:
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

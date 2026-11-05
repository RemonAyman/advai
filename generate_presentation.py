import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml import parse_xml

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank layout

    # Premium Professional Color Palette (Modern Medical Tech Theme)
    DARK_NAVY  = RGBColor(10, 17, 40)      # #0A1128 - Deepest dark hero
    NAVY_CARD  = RGBColor(15, 23, 42)      # #0F172A - Dark slate container
    NAVY_TITLE = RGBColor(15, 23, 42)      # #0F172A - Dominant title text
    TEAL       = RGBColor(14, 116, 144)    # #0E7490 - Primary teal
    CYAN_ACC   = RGBColor(2, 132, 199)     # #0284C7 - Bright vibrant blue
    CYAN_LIGHT = RGBColor(56, 189, 248)    # #38BDF8 - Glowing cyan
    SLATE_BODY = RGBColor(71, 85, 105)     # #475569 - Secondary body text
    LIGHT_BG   = RGBColor(248, 250, 252)   # #F8FAFC - Ultra-clean background
    WHITE      = RGBColor(255, 255, 255)
    BORDER_COL = RGBColor(226, 232, 240)   # #E2E8F0 - Soft clean borders
    CRIMSON    = RGBColor(225, 29, 72)     # #E11D48 - Malignant / problem accent
    EMERALD    = RGBColor(16, 185, 129)    # #10B981 - Success / champion green
    AMBER      = RGBColor(217, 119, 6)     # #D97706 - Warning / golden badge
    GOLD_LIGHT = RGBColor(254, 240, 138)   # #FEF08A - Yellow pill

    def set_slide_background(slide, color):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_slide_transition(slide, trans_type="fade", direction="l"):
        """Injects native OpenXML slide transitions for smooth presentation playback"""
        try:
            if trans_type == "push":
                xml_str = f'<p:transition xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" spd="med"><p:push dir="{direction}"/></p:transition>'
            elif trans_type == "wipe":
                xml_str = f'<p:transition xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" spd="med"><p:wipe dir="{direction}"/></p:transition>'
            elif trans_type == "split":
                xml_str = '<p:transition xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" spd="med"><p:split orient="horz"/></p:transition>'
            else:
                xml_str = '<p:transition xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" spd="med"><p:fade/></p:transition>'
            trans_element = parse_xml(xml_str)
            slide.element.append(trans_element)
        except Exception:
            pass

    def add_header(slide, title_text, category_text="SUPERVISED MACHINE LEARNING PROJECT", accent_color=CYAN_ACC):
        # Category Eyebrow Pill Shape
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.35), Inches(3.6), Inches(0.32))
        pill.fill.solid()
        pill.fill.fore_color.rgb = RGBColor(224, 242, 254) # Light soft cyan
        pill.line.fill.background()
        tf_pill = pill.text_frame
        tf_pill.word_wrap = False
        tf_pill.margin_top = tf_pill.margin_bottom = tf_pill.margin_left = tf_pill.margin_right = 0
        p_pill = tf_pill.paragraphs[0]
        p_pill.text = f"  ●  {category_text.upper()}"
        p_pill.font.size = Pt(9.5)
        p_pill.font.bold = True
        p_pill.font.color.rgb = CYAN_ACC

        # Main Title Box
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.733), Inches(0.75))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p_title = tf.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(21)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY_TITLE

        # Sleek Horizontal Accent Bar
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.42), Inches(2.2), Inches(0.04))
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent_color
        bar.line.fill.background()

    # =========================================================================
    # SLIDE 1: HERO TITLE SLIDE (Glassmorphic Dark Theme with Eng Mathew Greeting)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1, DARK_NAVY)
    add_slide_transition(slide1, "fade")

    # Hero Main Card Container
    card = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(0.7), Inches(11.533), Inches(6.1))
    card.fill.solid()
    card.fill.fore_color.rgb = NAVY_CARD
    card.line.color.rgb = RGBColor(30, 58, 138)
    card.line.width = Pt(1.5)

    # Top Accent Stripe on Hero Card
    top_line = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(0.7), Inches(11.533), Inches(0.08))
    top_line.fill.solid()
    top_line.fill.fore_color.rgb = CYAN_LIGHT
    top_line.line.fill.background()

    # Text Frame inside Hero Card
    tb = slide1.shapes.add_textbox(Inches(1.4), Inches(1.0), Inches(10.533), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True

    # Eyebrow Category
    p_cat = tf.paragraphs[0]
    p_cat.text = "🎗️ ADVANCED AI & SUPERVISED LEARNING IN CLINICAL ONCOLOGY"
    p_cat.font.size = Pt(12)
    p_cat.font.bold = True
    p_cat.font.color.rgb = CYAN_LIGHT

    # Main Presentation Title
    p_title = tf.add_paragraph()
    p_title.text = "BREAST CANCER WISCONSIN (ORIGINAL) CLASSIFICATION"
    p_title.font.size = Pt(27)
    p_title.font.bold = True
    p_title.font.color.rgb = WHITE
    p_title.space_before = Pt(8)

    # Subtitle
    p_sub = tf.add_paragraph()
    p_sub.text = "A Rigorous Comparative Benchmark of Five Supervised Algorithms for Automated Biopsy Diagnosis"
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = RGBColor(147, 197, 253)
    p_sub.space_before = Pt(6)

    # Special Review Dedication for Eng Mathew
    p_ded = tf.add_paragraph()
    p_ded.text = "👨‍🏫 Academic Review Session: Dedicated & Prepared for Eng Mathew"
    p_ded.font.size = Pt(12)
    p_ded.font.bold = True
    p_ded.font.color.rgb = GOLD_LIGHT
    p_ded.space_before = Pt(12)

    # Team Members (6 members, David removed cleanly)
    p_team = tf.add_paragraph()
    p_team.text = "👥 Project Team: Remon Ayman (Lead) • Marcelino • Bishoy • Mario • Engy • Karen"
    p_team.font.size = Pt(12)
    p_team.font.bold = True
    p_team.font.color.rgb = RGBColor(226, 232, 240)
    p_team.space_before = Pt(8)

    # Pipeline summary tag
    p_pipe = tf.add_paragraph()
    p_pipe.text = "Rigorous Pipeline: Exploration • Clean Preprocessing • Outlier & Correlation EDA • 5-Fold Stratified CV • Zero Leakage"
    p_pipe.font.size = Pt(11)
    p_pipe.font.color.rgb = RGBColor(148, 163, 184)
    p_pipe.space_before = Pt(8)

    # 4 KPI Stat Badges on Slide 1 Bottom
    kpi_badges = [
        ("97.12%", "Peak Test Accuracy", EMERALD),
        ("95.83%", "Clinical Recall (Sens.)", CRIMSON),
        ("0.9961", "Highest ROC-AUC (ANN)", CYAN_LIGHT),
        ("Zero Leakage", "Strict Train-Only Scaler", AMBER)
    ]
    for k_idx, (val, lbl, col) in enumerate(kpi_badges):
        bx = Inches(1.4 + k_idx * 2.6)
        by = Inches(5.5)
        badge = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bx, by, Inches(2.4), Inches(0.95))
        badge.fill.solid()
        badge.fill.fore_color.rgb = RGBColor(30, 41, 59)
        badge.line.color.rgb = col
        badge.line.width = Pt(1.2)

        btf = badge.text_frame
        btf.word_wrap = True
        btf.margin_top = btf.margin_bottom = Inches(0.12)
        bp1 = btf.paragraphs[0]
        bp1.text = val
        bp1.font.size = Pt(16)
        bp1.font.bold = True
        bp1.font.color.rgb = col
        bp1.alignment = PP_ALIGN.CENTER

        bp2 = btf.add_paragraph()
        bp2.text = lbl
        bp2.font.size = Pt(9)
        bp2.font.color.rgb = RGBColor(203, 213, 225)
        bp2.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT & PROJECT OBJECTIVE
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2, LIGHT_BG)
    add_header(slide2, "1. Problem Statement & Project Objectives", "EXECUTIVE SUMMARY", CRIMSON)
    add_slide_transition(slide2, "push", "l")

    # Left Column: Problem (with Crimson Accent Stripe)
    box_l = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.4))
    box_l.fill.solid()
    box_l.fill.fore_color.rgb = WHITE
    box_l.line.color.rgb = BORDER_COL

    stripe_l = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(0.08))
    stripe_l.fill.solid()
    stripe_l.fill.fore_color.rgb = CRIMSON
    stripe_l.line.fill.background()

    tf_l = box_l.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_right = Inches(0.35)
    tf_l.margin_top = Inches(0.3)
    p = tf_l.paragraphs[0]
    p.text = "Clinical Problem Statement"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CRIMSON

    points_l = [
        "Breast cancer remains one of the most prevalent and life-threatening oncological conditions globally.",
        "Fine Needle Aspirate (FNA) biopsy cytology requires specialized pathology interpretation, which can be subjective, fatigue-prone, and time-intensive.",
        "False Negatives in cancer diagnosis are catastrophic, leading to delayed life-saving intervention and treatment.",
        "An objective, reproducible computer-aided diagnostic (CAD) system is required to empower clinicians with fast, highly reliable secondary screening."
    ]
    for pt in points_l:
        p = tf_l.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11.5)
        p.font.color.rgb = SLATE_BODY
        p.space_before = Pt(10)

    # Right Column: Objectives (with Emerald Accent Stripe)
    box_r = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.4))
    box_r.fill.solid()
    box_r.fill.fore_color.rgb = WHITE
    box_r.line.color.rgb = BORDER_COL

    stripe_r = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(0.08))
    stripe_r.fill.solid()
    stripe_r.fill.fore_color.rgb = EMERALD
    stripe_r.line.fill.background()

    tf_r = box_r.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = Inches(0.35)
    tf_r.margin_top = Inches(0.3)
    p = tf_r.paragraphs[0]
    p.text = "Core Project Objectives"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    points_r = [
        "Rigorous Data Preprocessing: Impute missing values, remove duplicates, drop ID features, and eliminate all data leakage via strict train-only parameter fitting.",
        "In-Depth EDA: Uncover cytological feature distributions, multicollinearity, and domain outliers across 6 comprehensive visual analyses.",
        "Multi-Model Implementation: Implement and tune 5 distinct supervised classification paradigms: KNN, Decision Tree, Random Forest, ANN, and SVM.",
        "5-Fold Stratified Cross-Validation: Ensure hyperparameter tuning is scientifically grounded on training data without test contamination.",
        "Clinical Evaluation: Maximize Recall (Sensitivity) for malignant cases while evaluating Accuracy, Precision, F1-Score, and ROC-AUC."
    ]
    for pt in points_r:
        p = tf_r.add_paragraph()
        p.text = "✔ " + pt
        p.font.size = Pt(11.5)
        p.font.color.rgb = SLATE_BODY
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 3: DATASET SOURCE & ATTRIBUTES
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3, LIGHT_BG)
    add_header(slide3, "2. Dataset Source & Attribute Breakdown", "TASK 1: DATASET EXPLORATION", CYAN_ACC)
    add_slide_transition(slide3, "wipe", "r")

    box3 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.4))
    box3.fill.solid()
    box3.fill.fore_color.rgb = WHITE
    box3.line.color.rgb = BORDER_COL

    stripe3 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(0.08))
    stripe3.fill.solid()
    stripe3.fill.fore_color.rgb = CYAN_ACC
    stripe3.line.fill.background()

    tf3 = box3.text_frame
    tf3.word_wrap = True
    tf3.margin_left = tf3.margin_right = Inches(0.35)
    tf3.margin_top = Inches(0.25)

    p = tf3.paragraphs[0]
    p.text = "Dataset: Breast Cancer Wisconsin (Original) — UCI Machine Learning Repository"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    p = tf3.add_paragraph()
    p.text = "Donated by Dr. William H. Wolberg (University of Wisconsin Hospitals, Madison). Total 699 clinical instances, 11 attributes."
    p.font.size = Pt(12)
    p.font.color.rgb = SLATE_BODY
    p.space_before = Pt(4)

    attrs = [
        ("1. Sample_Code_Number", "Arbitrary Patient Identifier", "Dropped during preprocessing to prevent spurious correlation."),
        ("2. Clump_Thickness (1-10)", "Degree of cell layer agglomeration", "Higher values represent multilayered malignant clusters."),
        ("3. Uniformity_of_Cell_Size (1-10)", "Consistency in biopsy cell dimensions", "Severe irregularity is a classic hallmark of malignancy."),
        ("4. Uniformity_of_Cell_Shape (1-10)", "Cellular membrane geometry", "High correlation (r=0.91) with cell size uniformity."),
        ("5. Marginal_Adhesion (1-10)", "Inter-cellular adhesive strength", "Loss of adhesion facilitates metastatic invasion."),
        ("6. Single_Epithelial_Cell_Size (1-10)", "Cytoplasmic volume & nuclear enlargement", "Hypertrophy indicates malignant transformation."),
        ("7. Bare_Nuclei (1-10)", "Nuclei unencapsulated by cytoplasm", "Contains 16 missing values ('?'); vital predictor (r=0.82)."),
        ("8. Bland_Chromatin (1-10)", "Nuclear chromatin texture", "Coarse chromatin aggregates distinguish cancer cells."),
        ("9. Normal_Nucleoli (1-10)", "Nucleolar prominence and number", "Hyperactive nucleoli reflect rapid malignant cell division."),
        ("10. Mitoses (1-10)", "Observed mitotic figures in cytology", "Direct indicator of cellular proliferation rate."),
        ("11. Class (Target Variable)", "2 = Benign (65.5%), 4 = Malignant (34.5%)", "Re-mapped to 0 (Benign) and 1 (Malignant) as positive class.")
    ]

    for name, desc, note in attrs:
        p = tf3.add_paragraph()
        p.text = f"{name}: {desc} — {note}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = CRIMSON if "Class" in name else SLATE_BODY
        p.font.bold = True if "Class" in name else False
        p.space_before = Pt(2.5)

    # =========================================================================
    # SLIDE 4: DATA PREPROCESSING & LEAKAGE PREVENTION
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4, LIGHT_BG)
    add_header(slide4, "3. Data Cleaning, Preprocessing & Leakage Prevention", "TASK 2: DATA PREPROCESSING", TEAL)
    add_slide_transition(slide4, "push", "l")

    steps = [
        ("Handling Missing Values", "16 records (2.3%) in 'Bare_Nuclei' held '?' strings. Replaced with NaN. Imputed via Median = 1.0. Imputer was FIT strictly on X_train only to prevent test contamination."),
        ("Handling Duplicate Records", "Identified 8 exact duplicate records (identical ID and identical features). Removed to preserve independent observation integrity, retaining 691 clean records."),
        ("Removing Irrelevant Features", "Dropped 'Sample_Code_Number'. Unique patient IDs contain no clinical or biological predictive signal and cause artificial memorization."),
        ("Target Class Re-mapping", "Mapped Benign (2) -> 0 and Malignant (4) -> 1. Formatted malignant cases explicitly as the positive class as demanded by medical diagnostic protocol."),
        ("Stratified 80:20 Partition", "Train set = 552 samples (80%), Test set = 139 samples (20%). Stratified split guarantees exactly 34.5% malignant cases in both splits."),
        ("Standardization Without Leakage", "StandardScaler fitted on X_train only; mean and variance parameters applied to transform X_train and unseen X_test.")
    ]

    for idx, (title, text) in enumerate(steps):
        row = idx // 2
        col = idx % 2
        x = Inches(0.8 + col * 6.0)
        y = Inches(1.6 + row * 1.8)
        
        card_s4 = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(5.6), Inches(1.65))
        card_s4.fill.solid()
        card_s4.fill.fore_color.rgb = WHITE
        card_s4.line.color.rgb = BORDER_COL

        # Top mini indicator
        mini_stripe = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(5.6), Inches(0.06))
        mini_stripe.fill.solid()
        mini_stripe.fill.fore_color.rgb = TEAL if idx % 2 == 0 else CYAN_ACC
        mini_stripe.line.fill.background()

        tf = card_s4.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.2)
        tf.margin_top = Inches(0.16)

        p = tf.paragraphs[0]
        p.text = f"Step {idx+1}: {title}"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = TEAL

        p2 = tf.add_paragraph()
        p2.text = text
        p2.font.size = Pt(10)
        p2.font.color.rgb = SLATE_BODY
        p2.space_before = Pt(3)

    # =========================================================================
    # SLIDE 5: EDA - MISSING VALUES & TARGET DISTRIBUTION
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5, LIGHT_BG)
    add_header(slide5, "4. EDA: Missing Values & Target Class Distribution", "TASK 3: EXPLORATORY DATA ANALYSIS", CYAN_ACC)
    add_slide_transition(slide5, "fade")

    if os.path.exists("figures/1_missing_values.png"):
        slide5.shapes.add_picture("figures/1_missing_values.png", Inches(0.8), Inches(1.6), width=Inches(5.7))
    if os.path.exists("figures/2_target_distribution.png"):
        slide5.shapes.add_picture("figures/2_target_distribution.png", Inches(6.8), Inches(1.6), width=Inches(5.7))

    card5 = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8), Inches(11.733), Inches(1.2))
    card5.fill.solid()
    card5.fill.fore_color.rgb = WHITE
    card5.line.color.rgb = BORDER_COL

    tf5 = card5.text_frame
    tf5.word_wrap = True
    tf5.margin_left = tf5.margin_right = Inches(0.25)
    tf5.margin_top = Inches(0.12)
    p = tf5.paragraphs[0]
    p.text = "Key Clinical & Statistical Insights:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    p2 = tf5.add_paragraph()
    p2.text = "• Missingness is isolated to Bare_Nuclei (16 records / 2.3%). Median imputation preserves all samples without statistical distortion.\n• Class distribution: 453 Benign (65.6%) vs 238 Malignant (34.4%). Imbalance is moderate and naturally handled by stratified sampling."
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = SLATE_BODY

    # =========================================================================
    # SLIDE 6: EDA - HISTOGRAMS & OUTLIER BOXPLOTS
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6, LIGHT_BG)
    add_header(slide6, "5. EDA: Feature Distributions & Outlier Analysis", "TASK 3: EXPLORATORY DATA ANALYSIS", CYAN_ACC)
    add_slide_transition(slide6, "push", "l")

    if os.path.exists("figures/3_feature_histograms.png"):
        slide6.shapes.add_picture("figures/3_feature_histograms.png", Inches(0.8), Inches(1.6), width=Inches(5.7))
    if os.path.exists("figures/4_feature_boxplots.png"):
        slide6.shapes.add_picture("figures/4_feature_boxplots.png", Inches(6.8), Inches(1.6), width=Inches(5.7))

    card6 = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8), Inches(11.733), Inches(1.2))
    card6.fill.solid()
    card6.fill.fore_color.rgb = WHITE
    card6.line.color.rgb = BORDER_COL

    tf6 = card6.text_frame
    tf6.word_wrap = True
    tf6.margin_left = tf6.margin_right = Inches(0.25)
    tf6.margin_top = Inches(0.12)
    p = tf6.paragraphs[0]
    p.text = "Key Clinical & Statistical Insights:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    p2 = tf6.add_paragraph()
    p2.text = "• Histograms reveal pronounced right-skewness: normal cells concentrate heavily at score 1. Scores > 3 represent rare malignant anomalies.\n• Boxplots show statistical upper outliers in Mitoses and Adhesion; in oncology, these are true pathological signals, NOT noisy outliers to remove."
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = SLATE_BODY

    # =========================================================================
    # SLIDE 7: EDA - CORRELATION HEATMAP
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7, LIGHT_BG)
    add_header(slide7, "6. EDA: Correlation Heatmap & Feature Interrelationships", "TASK 3: EXPLORATORY DATA ANALYSIS", CYAN_ACC)
    add_slide_transition(slide7, "fade")

    if os.path.exists("figures/5_correlation_heatmap.png"):
        slide7.shapes.add_picture("figures/5_correlation_heatmap.png", Inches(0.8), Inches(1.6), width=Inches(5.6))

    box7_r = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.7), Inches(1.6), Inches(5.8), Inches(5.3))
    box7_r.fill.solid()
    box7_r.fill.fore_color.rgb = WHITE
    box7_r.line.color.rgb = BORDER_COL

    stripe7 = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.7), Inches(1.6), Inches(5.8), Inches(0.08))
    stripe7.fill.solid()
    stripe7.fill.fore_color.rgb = CYAN_ACC
    stripe7.line.fill.background()

    tf7 = box7_r.text_frame
    tf7.word_wrap = True
    tf7.margin_left = tf7.margin_right = Inches(0.35)
    tf7.margin_top = Inches(0.25)

    p = tf7.paragraphs[0]
    p.text = "Correlation & Multicollinearity Findings"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    points7 = [
        "Strongest Linear Malignancy Indicators: Uniformity_Cell_Size (r = 0.82), Uniformity_Cell_Shape (r = 0.82), and Bare_Nuclei (r = 0.82) show the highest positive correlation with cancer diagnosis.",
        "Severe Multicollinearity (r = 0.91): Uniformity of Cell Size and Cell Shape correlate at 0.91. This occurs because malignant cells simultaneously lose size uniformity and structural symmetry.",
        "Algorithmic Sensitivity to Collinearity: Collinearity can degrade linear model coefficient stability; however, tree ensembles (Random Forest) and non-linear kernel SVMs handle collinear feature spaces with high resilience.",
        "Mitoses Predictor: Exhibits lower correlation (r = 0.42) because mitotic division occurs in bursts, but remains an essential feature for identifying aggressive carcinomas."
    ]
    for pt in points7:
        p = tf7.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11)
        p.font.color.rgb = SLATE_BODY
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 8: STANDARDIZATION BEFORE VS AFTER
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8, LIGHT_BG)
    add_header(slide8, "7. Feature Standardization: Before vs. After Analysis", "TASK 3: EXPLORATORY DATA ANALYSIS", TEAL)
    add_slide_transition(slide8, "wipe", "r")

    if os.path.exists("figures/6_before_after_standardization.png"):
        slide8.shapes.add_picture("figures/6_before_after_standardization.png", Inches(0.8), Inches(1.6), width=Inches(11.733))

    card8 = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8), Inches(11.733), Inches(1.2))
    card8.fill.solid()
    card8.fill.fore_color.rgb = WHITE
    card8.line.color.rgb = BORDER_COL

    tf8 = card8.text_frame
    tf8.word_wrap = True
    tf8.margin_left = tf8.margin_right = Inches(0.25)
    tf8.margin_top = Inches(0.12)
    p = tf8.paragraphs[0]
    p.text = "Standardization Justification & Mathematical Impact:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    p2 = tf8.add_paragraph()
    p2.text = "• Before: Features have differing variances despite uniform [1, 10] range, creating distance metric distortion in KNN and SVM.\n• After: Z-score standardization shifts mean to 0 and variance to 1. Ensures equal contribution to Euclidean distances in KNN, equal margin penalties in SVM, and balanced gradient backpropagation in ANN."
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = SLATE_BODY

    # =========================================================================
    # SLIDE 9: FIVE ALGORITHMS & CROSS-VALIDATION
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9, LIGHT_BG)
    add_header(slide9, "8. Model Implementation & Stratified 5-Fold Cross-Validation", "TASK 4: MODEL IMPLEMENTATION", CYAN_ACC)
    add_slide_transition(slide9, "split")

    algos = [
        ("K-Nearest Neighbors (KNN)", "Instance-based memory classifier. Classifies unseen samples via majority vote of k nearest Euclidean neighbors.", "Optimal: k=7, weights='uniform', metric='euclidean'"),
        ("Decision Tree (DT)", "Greedy non-parametric partitioner creating orthogonal decision boundaries based on information gain/entropy.", "Optimal: criterion='gini', max_depth=4, min_samples_split=2"),
        ("Random Forest (RF)", "Ensemble of 100 bagged de-correlated decision trees with random feature subspacing to eliminate single-tree variance.", "Optimal: n_estimators=100, max_depth=3, min_samples_split=10"),
        ("Artificial Neural Network (ANN)", "Multi-Layer Perceptron (MLP) with non-linear activation, hidden layer transformations, and L2 weight decay.", "Optimal: hidden_layers=(32,), activation='relu', alpha=0.1"),
        ("Support Vector Machine (SVM)", "Maximum margin hyperplane separator employing Radial Basis Function (RBF) kernel mapping for non-linear separation.", "Optimal: kernel='rbf', C=0.1, gamma='scale'")
    ]

    for idx, (name, desc, opt) in enumerate(algos):
        y_pos = Inches(1.6 + idx * 1.05)
        bar = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y_pos, Inches(11.733), Inches(0.95))
        bar.fill.solid()
        bar.fill.fore_color.rgb = WHITE
        bar.line.color.rgb = BORDER_COL

        # Left mini border stripe
        left_accent = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y_pos, Inches(0.12), Inches(0.95))
        left_accent.fill.solid()
        left_accent.fill.fore_color.rgb = CYAN_ACC if idx % 2 == 0 else TEAL
        left_accent.line.fill.background()

        tf = bar.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.25)
        tf.margin_top = Inches(0.12)

        p = tf.paragraphs[0]
        p.text = f"{name}: "
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = TEAL

        run_desc = p.add_run()
        run_desc.text = desc + " "
        run_desc.font.size = Pt(10.5)
        run_desc.font.bold = False
        run_desc.font.color.rgb = SLATE_BODY

        run_opt = p.add_run()
        run_opt.text = f"[{opt}]"
        run_opt.font.size = Pt(10.5)
        run_opt.font.bold = True
        run_opt.font.color.rgb = NAVY_TITLE

    # =========================================================================
    # SLIDE 10: CONFUSION MATRICES
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10, LIGHT_BG)
    add_header(slide10, "9. Model Evaluation: Confusion Matrices (Test N=139)", "TASK 5: MODEL EVALUATION", CRIMSON)
    add_slide_transition(slide10, "fade")

    if os.path.exists("figures/7_all_confusion_matrices.png"):
        slide10.shapes.add_picture("figures/7_all_confusion_matrices.png", Inches(0.8), Inches(1.5), width=Inches(7.2))

    box10_r = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(1.5), Inches(4.2), Inches(5.4))
    box10_r.fill.solid()
    box10_r.fill.fore_color.rgb = WHITE
    box10_r.line.color.rgb = BORDER_COL

    stripe10 = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(1.5), Inches(4.2), Inches(0.08))
    stripe10.fill.solid()
    stripe10.fill.fore_color.rgb = CRIMSON
    stripe10.line.fill.background()

    tf10 = box10_r.text_frame
    tf10.word_wrap = True
    tf10.margin_left = tf10.margin_right = Inches(0.3)
    tf10.margin_top = Inches(0.25)

    p = tf10.paragraphs[0]
    p.text = "Confusion Matrix Analysis"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    points10 = [
        "Test Sample Split: Exactly 91 Benign (Class 0) and 48 Malignant (Class 1) cases.",
        "Clinical Priority: Minimizing False Negatives (FN = Malignant classified as Benign). Missing cancer delays vital medical care.",
        "Top Trio (RF, ANN, SVM): Each achieved only 2 False Negatives and 2 False Positives across 139 patients, yielding 95.83% Recall.",
        "KNN: Produced 4 False Negatives (Recall 91.67%) and only 1 False Positive (Precision 97.78%).",
        "Decision Tree: Lagged with 5 False Negatives and 5 False Positives (Accuracy 92.81%)."
    ]
    for pt in points10:
        p = tf10.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10.5)
        p.font.color.rgb = SLATE_BODY
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 11: COMPARISON TABLE & ROC CURVES
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11, LIGHT_BG)
    add_header(slide11, "10. Performance Comparison & ROC Curves", "TASK 5: MODEL EVALUATION", CYAN_ACC)
    add_slide_transition(slide11, "push", "l")

    if os.path.exists("figures/8_roc_curves_comparison.png"):
        slide11.shapes.add_picture("figures/8_roc_curves_comparison.png", Inches(0.8), Inches(1.6), width=Inches(5.5))

    # Add Table of results
    table_shape = slide11.shapes.add_table(6, 6, Inches(6.6), Inches(1.6), Inches(5.9), Inches(3.2))
    table = table_shape.table

    cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    for c_idx, col_name in enumerate(cols):
        cell = table.cell(0, c_idx)
        cell.text = col_name
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY_TITLE
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER

    data_rows = [
        ("KNN", "96.40%", "97.78%", "91.67%", "0.9462", "0.9823"),
        ("Decision Tree", "92.81%", "89.58%", "89.58%", "0.8958", "0.9755"),
        ("Random Forest ⭐", "97.12%", "95.83%", "95.83%", "0.9583", "0.9929"),
        ("ANN (MLP) ⭐", "97.12%", "95.83%", "95.83%", "0.9583", "0.9961"),
        ("SVM (RBF) ⭐", "97.12%", "95.83%", "95.83%", "0.9583", "0.9945")
    ]

    for r_idx, row in enumerate(data_rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            if r_idx in [2, 3, 4]:
                cell.fill.fore_color.rgb = RGBColor(236, 253, 245) # Soft Emerald Tint
            else:
                cell.fill.fore_color.rgb = WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = EMERALD if r_idx in [2, 3, 4] and c_idx > 0 else NAVY_TITLE
                p.font.bold = True if r_idx in [2, 3, 4] else False
                p.alignment = PP_ALIGN.CENTER

    # Note card below table
    card11 = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.6), Inches(5.1), Inches(5.9), Inches(1.8))
    card11.fill.solid()
    card11.fill.fore_color.rgb = WHITE
    card11.line.color.rgb = BORDER_COL

    tf11 = card11.text_frame
    tf11.word_wrap = True
    tf11.margin_left = tf11.margin_right = Inches(0.2)
    tf11.margin_top = Inches(0.15)
    p = tf11.paragraphs[0]
    p.text = "Key Findings:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    p2 = tf11.add_paragraph()
    p2.text = "• RF, ANN, and SVM tie for top Accuracy (97.12%) and Recall (95.83%).\n• ANN achieves the highest overall ROC-AUC (0.9961), closely followed by SVM (0.9945) and Random Forest (0.9929).\n• All models significantly outperform baseline guessing."
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = SLATE_BODY

    # =========================================================================
    # SLIDE 12: BEST PERFORMING MODEL & JUSTIFICATION
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide12, LIGHT_BG)
    add_header(slide12, "11. Best-Performing Model & Justification", "TASK 5: MODEL SELECTION", EMERALD)
    add_slide_transition(slide12, "push", "l")

    if os.path.exists("figures/10_rf_feature_importance.png"):
        slide12.shapes.add_picture("figures/10_rf_feature_importance.png", Inches(0.8), Inches(1.6), width=Inches(5.5))

    box12_r = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.6), Inches(1.6), Inches(5.9), Inches(5.3))
    box12_r.fill.solid()
    box12_r.fill.fore_color.rgb = WHITE
    box12_r.line.color.rgb = EMERALD
    box12_r.line.width = Pt(2.5)

    stripe12 = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.6), Inches(1.6), Inches(5.9), Inches(0.08))
    stripe12.fill.solid()
    stripe12.fill.fore_color.rgb = EMERALD
    stripe12.line.fill.background()

    tf12 = box12_r.text_frame
    tf12.word_wrap = True
    tf12.margin_left = tf12.margin_right = Inches(0.35)
    tf12.margin_top = Inches(0.25)

    p = tf12.paragraphs[0]
    p.text = "🏆 Selected Champion: Random Forest (RF)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    justifications = [
        "Peak Clinical Recall (95.83%) & Accuracy (97.12%): Successfully identified 46 out of 48 malignant cases with minimal misdiagnosis.",
        "Clinical Explainability & Trust: Unlike black-box neural networks or high-dimensional SVM kernels, Random Forest provides transparent Gini feature importance.",
        "Pathological Validation: Identified Uniformity_Cell_Size, Bare_Nuclei, and Uniformity_Cell_Shape as primary drivers, matching published cytology literature.",
        "Overfitting Resistance: Bagging 100 de-correlated trees stabilizes predictions against biopsy noise and physician score variance.",
        "Computational Simplicity & Zero Tuning Drift: Instant training and execution without vulnerability to local minima or vanishing gradients."
    ]

    for j in justifications:
        p = tf12.add_paragraph()
        p.text = "✔ " + j
        p.font.size = Pt(11)
        p.font.color.rgb = SLATE_BODY
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 13: CONCLUSION & FUTURE WORK WITH SCAN-READY QR CODE
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide13, LIGHT_BG)
    add_header(slide13, "12. Project Conclusion & Future Improvements", "PROJECT WRAP-UP", TEAL)
    add_slide_transition(slide13, "fade")

    # Column 1: Conclusions
    box13_l = slide13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(4.5), Inches(4.5))
    box13_l.fill.solid()
    box13_l.fill.fore_color.rgb = WHITE
    box13_l.line.color.rgb = BORDER_COL

    stripe13_l = slide13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(4.5), Inches(0.08))
    stripe13_l.fill.solid()
    stripe13_l.fill.fore_color.rgb = TEAL
    stripe13_l.line.fill.background()

    tf13_l = box13_l.text_frame
    tf13_l.word_wrap = True
    tf13_l.margin_left = tf13_l.margin_right = Inches(0.25)
    tf13_l.margin_top = Inches(0.2)
    p = tf13_l.paragraphs[0]
    p.text = "Project Conclusions"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    concs = [
        "Demonstrated successful end-to-end implementation of a supervised classification system for breast cytology diagnosis.",
        "Strict prevention of data leakage ensured that test evaluation metrics represent true prospective clinical performance.",
        "Supervised machine learning algorithms achieve over 97% accuracy and 95.8% sensitivity in distinguishing benign from malignant biopsies.",
        "Automated classification provides a robust, fast, and objective second opinion for clinical oncologists, reducing diagnostic error."
    ]
    for c in concs:
        p = tf13_l.add_paragraph()
        p.text = "• " + c
        p.font.size = Pt(10.5)
        p.font.color.rgb = SLATE_BODY
        p.space_before = Pt(6)

    # Column 2: Future Work
    box13_m = slide13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.5), Inches(1.6), Inches(4.5), Inches(4.5))
    box13_m.fill.solid()
    box13_m.fill.fore_color.rgb = WHITE
    box13_m.line.color.rgb = BORDER_COL

    stripe13_m = slide13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.5), Inches(1.6), Inches(4.5), Inches(0.08))
    stripe13_m.fill.solid()
    stripe13_m.fill.fore_color.rgb = CYAN_ACC
    stripe13_m.line.fill.background()

    tf13_m = box13_m.text_frame
    tf13_m.word_wrap = True
    tf13_m.margin_left = tf13_m.margin_right = Inches(0.25)
    tf13_m.margin_top = Inches(0.2)
    p = tf13_m.paragraphs[0]
    p.text = "Future Improvements"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACC

    futures = [
        "Cost-Sensitive Loss Optimization: Incorporate asymmetric misclassification costs to push False Negatives to exactly zero.",
        "Soft-Voting Ensemble Integration: Construct a meta-ensemble combining Random Forest, SVM, and ANN class probability predictions.",
        "Explainable AI (XAI) with SHAP: Implement SHapley Additive exPlanations to visualize per-patient cytological risk attribution.",
        "Computer Vision & Deep Learning: Train CNNs on raw digitised FNA microscopy biopsy images (WDBC/histopathology) to bypass manual feature scoring."
    ]
    for f in futures:
        p = tf13_m.add_paragraph()
        p.text = "➔ " + f
        p.font.size = Pt(10.5)
        p.font.color.rgb = SLATE_BODY
        p.space_before = Pt(6)

    # Column 3: Live Demo Card with Scannable QR Code
    box13_qr = slide13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.2), Inches(1.6), Inches(2.333), Inches(4.5))
    box13_qr.fill.solid()
    box13_qr.fill.fore_color.rgb = WHITE
    box13_qr.line.color.rgb = CYAN_ACC
    box13_qr.line.width = Pt(1.5)

    stripe13_qr = slide13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.2), Inches(1.6), Inches(2.333), Inches(0.08))
    stripe13_qr.fill.solid()
    stripe13_qr.fill.fore_color.rgb = EMERALD
    stripe13_qr.line.fill.background()

    tf13_qr = box13_qr.text_frame
    tf13_qr.word_wrap = True
    tf13_qr.margin_left = tf13_qr.margin_right = Inches(0.12)
    tf13_qr.margin_top = Inches(0.15)

    p_qr_t = tf13_qr.paragraphs[0]
    p_qr_t.text = "📱 Live Demo App"
    p_qr_t.font.size = Pt(12)
    p_qr_t.font.bold = True
    p_qr_t.font.color.rgb = NAVY_TITLE
    p_qr_t.alignment = PP_ALIGN.CENTER

    p_qr_sub = tf13_qr.add_paragraph()
    p_qr_sub.text = "Scan with your phone to test live right now:"
    p_qr_sub.font.size = Pt(8.5)
    p_qr_sub.font.color.rgb = SLATE_BODY
    p_qr_sub.alignment = PP_ALIGN.CENTER
    p_qr_sub.space_before = Pt(2)

    # Insert QR Code Image
    if os.path.exists("figures/app_qr_code.png"):
        slide13.shapes.add_picture("figures/app_qr_code.png", Inches(10.5), Inches(2.45), width=Inches(1.75), height=Inches(1.75))

    p_qr_url = tf13_qr.add_paragraph()
    p_qr_url.text = "advaiproject.streamlit.app"
    p_qr_url.font.size = Pt(9)
    p_qr_url.font.bold = True
    p_qr_url.font.color.rgb = CYAN_ACC
    p_qr_url.alignment = PP_ALIGN.CENTER
    p_qr_url.space_before = Pt(135) # Space past the image

    # Bottom Repository & Deployment Bar
    repo_bar = slide13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.3), Inches(11.733), Inches(0.7))
    repo_bar.fill.solid()
    repo_bar.fill.fore_color.rgb = NAVY_CARD
    repo_bar.line.color.rgb = CYAN_LIGHT
    repo_bar.line.width = Pt(1.0)

    tf_bar = repo_bar.text_frame
    tf_bar.word_wrap = True
    tf_bar.margin_top = Inches(0.12)
    p_b = tf_bar.paragraphs[0]
    p_b.text = "🚀 Live Web App Ready: https://advaiproject.streamlit.app   |   📂 Open Source Codebase: https://github.com/RemonAyman/advai"
    p_b.font.size = Pt(11)
    p_b.font.bold = True
    p_b.font.color.rgb = CYAN_LIGHT
    p_b.alignment = PP_ALIGN.CENTER

    prs.save("breast_cancer_classification_presentation.pptx")
    print("Successfully generated breast_cancer_classification_presentation.pptx (13 upgraded animated slides)!")

if __name__ == '__main__':
    create_deck()

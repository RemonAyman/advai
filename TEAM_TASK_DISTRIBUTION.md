# 👥 تقرير توزيع مهام المشروع بالتساوي على أعضاء الفريق (6 أشخاص)
## Project Task & Workload Distribution Matrix (Fair & Equal Division)

**اسم المشروع**: Supervised Machine Learning Classification Project — Breast Cancer Wisconsin (Original)  
**المستودع (Repository)**: [RemonAyman/advai](https://github.com/RemonAyman/advai)  
**عدد أعضاء الفريق**: 6 أعضاء  
**نسبة مساهمة كل عضو**: **16.67% لكل عضو بالتساوي وبنفس مستوى الصعوبة التقنية والجهد**

---

### 📊 نظرة عامة على مصفوفة التوزيع (Executive Workload Matrix)

| العضو (Member) | المسمى التقني في المشروع (Role) | التاسك المسؤول عنه (Assigned Task) | الملفات والكود الخاص به (Files & Code Owned) | شرائح العرض في البريزنتيشن (Slides) | نسبة الجهد |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **1. ريمون أيمن (Remon Ayman)** | **Team Lead & Preprocessing Architect** | **Task 2**: Data Preprocessing, Leakage Prevention & CLI | `run_project.py` (Prep), `predict_cli.py`, `run_app.bat`, `imputer.joblib`, `scaler.joblib` | Slides 1, 2, 4 | **16.67%** |
| **2. مارسلينو (Marcelino)** | **Exploratory Data Analysis Specialist (Part 1)** | **Task 1** & **Task 3**: Dataset Exploration & Distributions | `figures/1_missing_values.png`, `figures/2_target_distribution.png`, `figures/3_feature_histograms.png` | Slides 3, 5, 6 | **16.67%** |
| **3. بيشوى (Bishoy)** | **Advanced EDA & Cytology Relationships Specialist** | **Task 3**: Outliers, Correlation & Standardization | `figures/4_feature_boxplots.png`, `figures/5_correlation_heatmap.png`, `figures/6_before_after_standardization.png` | Slides 6, 7, 8 | **16.67%** |
| **4. ماريو (Mario)** | **Supervised ML Engineer (Instance & Tree Models)** | **Task 4**: KNN & Decision Tree Modeling & Tuning | `models/knn.joblib`, `models/decision_tree.joblib` | Slide 9 (Part 1) | **16.67%** |
| **5. إنجي (Engy)** | **Supervised ML Engineer (Neural & Kernel Models)** | **Task 4**: ANN (Neural Networks) & SVM Modeling & Tuning | `models/artificial_neural_network_ann.joblib`, `models/support_vector_machine_svm.joblib` | Slide 9 (Part 2) | **16.67%** |
| **6. كارين (Karen)** | **Ensemble Learning, Benchmark & Web App Specialist** | **Task 4, 5 & App**: Random Forest, Voting Ensemble, Metrics, Benchmark & Streamlit App | `models/random_forest.joblib`, `models/ensemble_voting.joblib`, `app.py`, `figures/7, 8, 9, 10` | Slides 10, 11, 12, 13 | **16.67%** |

---

## 🔍 التفاصيل الدقيقة لمسؤولية كل عضو (Detailed Breakdown)

---

### 1. ريمون أيمن (Remon Ayman) — مهندس معالجة البيانات ومنع التسرب والواجهة السريعة (Team Lead & Preprocessing Architect)
* **نسبة المشاركة**: 16.67%
* **التاسك الرئيسي**: **Task 2: Data Preprocessing, Leakage Prevention & CLI Tool**
* **المهام التقنية بالتفصيل**:
  1. فحص القيم المفقودة مشفرة كعلامة استفهام `'?'` في متغير `Bare_Nuclei` واستبدالها بـ `NaN`.
  2. تطبيق استراتيجية التعويض بالوسيط (`SimpleImputer(strategy='median')`) والتأكد بنسبة 100% أن التدريب يتم على `X_train` فقط لمنع الـ Data Leakage.
  3. رصد وحذف الـ 8 صفوف المكررة بالكامل (نفس المعرف ونفس الخصائص) للحفاظ على استقلالية العينات.
  4. حذف المعرف الفريد `Sample_Code_Number` لمنع النموذج من حفظ أرقام العينات عشوائياً.
  5. تقسيم البيانات بنسبة 80% تدريب (552 عينة) و 20% اختبار (139 عينة) بنظام التوزيع الطبقي (`stratify=y`).
  6. تدريب وتطبيق `StandardScaler` على بيانات التدريب ثم تحويل بيانات الاختبار بنفس المعايير ($\mu, \sigma$).
  7. بناء أداة التشخيص السريع من سطر الأوامر [`predict_cli.py`](file:///e:/advai/predict_cli.py) وسكريبت التشغيل المباشر [`run_app.bat`](file:///e:/advai/run_app.bat).
* **الملفات المسلمة من قبله**: مسار Preprocessing في `run_project.py`، `models/imputer.joblib`، `models/scaler.joblib`، `predict_cli.py`، `run_app.bat`.
* **الشرائح التي يشرحها في البريزنتيشن**:
  * **Slide 1**: عنوان المشروع والمقدمة وأعضاء الفريق.
  * **Slide 2**: المشكلة السريرية (Problem Statement) والأهداف الأساسية.
  * **Slide 4**: خطوات تنظيف البيانات بالتفصيل وشرح كيفية منع تسرب البيانات (Data Leakage).
* **سؤال المناقشة المتوقع **: *"إزاي تأكدت إن الـ StandardScaler والـ Imputer معملوش تسريب داتا من الـ Test للـ Train؟"*  
  **الإجابة النموذجية**: *"عملنا fit على X_train فقط، واحتفظنا ببيانات الـ Test معزولة تماماً وطبقنا عليها transform فقط بالقيم اللي اتعلمها من التدريب."*

---

### 2. مارسلينو (Marcelino) — أخصائي استكشاف البيانات والتوزيعات (EDA Specialist - Part 1)
* **نسبة المشاركة**: 16.67%
* **التاسك الرئيسي**: **Task 1: Dataset Exploration & Task 3: Distribution Analysis**
* **المهام التقنية بالتفصيل**:
  1. استكشاف واستيراد الداتا سيت الأصلية وتحليل الأبعاد الأولية (699 صف و 11 عمود).
  2. دراسة أنواع البيانات وتحديد مشكلة عمود `Bare_Nuclei` الذي تحول تلقائياً إلى نوع نصي Object.
  3. توليد وتصميم الرسم البياني للقيم المفقودة وتوثيقه (`figures/1_missing_values.png`).
  4. دراسة توزيع فئات الهدف وحساب نسبة الأورام الحميدة (65.5%) والخبيثة (34.5%) ورسم المخطط المزدوج الدائري والعمودي (`figures/2_target_distribution.png`).
  5. رسم وتوليد المدرجات التكرارية (Histograms) للخصائص الخلوية التسعة ومقارنة توزيعات الحميد والخبيث بداخلها (`figures/3_feature_histograms.png`).
* **الملفات المسلمة من قبله**: كود استكشاف البيانات في `run_project.py` والرسومات البيانية `figures/1_missing_values.png` و `figures/2_target_distribution.png` و `figures/3_feature_histograms.png`.
* **الشرائح التي يشرحها في البريزنتيشن**:
  * **Slide 3**: مصدر الداتا سيت (UCI Wolberg) وأبعادها وخصائص المتغيرات التسعة.
  * **Slide 5**: تحليل نسب الأورام الحميدة والخبيثة وتوزيع البيانات الأولية.
  * **Slide 6 (الجزء الأول)**: قراءة الهستوجرام للخصائص الخلوية وكيف تفصل بين الأنسجة السليمة والمصابة.
* **سؤال المناقشة المتوقع **: *"ليه فضلتم التعويض بالـ Median على الـ Mean في عمود Bare_Nuclei؟"*  
  **الإجابة النموذجية**: *"لأن قيم الخصائص الخلوية كلها قيم صحيحة ومنفصلة (Ordinal Integers من 1 لـ 10) والبيانات فيها التواء، والـ Median بيحافظ على نوعية الرقم الصحيحة (1.0) ولا يتأثر بالقيم المتطرفة."*

---

### 3. بيشوى (Bishoy) — خبير العلاقات الخلوية والارتباط والتقييس (Advanced EDA Specialist)
* **نسبة المشاركة**: 16.67%
* **التاسك الرئيسي**: **Task 3: Outlier Detection, Correlation Analysis & Standardization Impact**
* **المهام التقنية بالتفصيل**:
  1. تحليل القيم المتطرفة (Outliers) وتوليد مخططات الصندوق (Boxplots) لجميع المتغيرات الخلوية التسعة (`figures/4_feature_boxplots.png`).
  2. التفسير البيولوجي للقيم الشاذة في متغير الانقسام الميتوزي (`Mitoses`) والتأكيد على أنها تعبير عن ورم شرس وليست أخطاء إدخال بيانات.
  3. حساب مصفوفة ارتباط بيرسون (Pearson Correlation Matrix) وتصميم الخريطة الحرارية (`figures/5_correlation_heatmap.png`).
  4. كشف ظاهرة الـ Multicollinearity بين حجم وشكل الخلية (`Uniformity_Cell_Size` و `Uniformity_Cell_Shape` بارتباط 0.90).
  5. تصميم مقارنة بصرية لتأثير التقييس المعياري على تشتت البيانات قبل وبعد StandardScaler (`figures/6_before_after_standardization.png`).
* **الملفات المسلمة من قبله**: موديول التحليل الإحصائي المتقدم في `run_project.py` والرسومات `figures/4_feature_boxplots.png`، `figures/5_correlation_heatmap.png`، `figures/6_before_after_standardization.png`.
* **الشرائح التي يشرحها في البريزنتيشن**:
  * **Slide 6 (الجزء الثاني)**: تفسير مخططات الـ Boxplots وأسباب الاحتفاظ بالقيم الشاذة وعدم حذفها.
  * **Slide 7**: قراءة خريطة الارتباط (Heatmap) وتحديد أقوى العوامل المسببة للورم الخبيث.
  * **Slide 8**: شرح لماذا تحتاج خوارزميات مثل KNN و SVM للتقييس المعياري (Standardization).
* **سؤال المناقشة المتوقع **: *"ليه محذفتوش الـ Outliers اللي ظهرت في متغير Mitoses؟"*  
  **الإجابة النموذجية**: *"لأن في الأورام السرطانية، معدل الانقسام الميتوزي المرتفع هو علامة بيولوجية مؤكدة للورم الخبيث الشرس (High-grade Malignancy)؛ وحذفه كان هيحرم النموذج من كشف أخطر الحالات."*

---

### 4. ماريو (Mario) — مهندس نماذج الجيران وأشجار القرار (Instance & Tree Models Engineer)
* **نسبة المشاركة**: 16.67%
* **التاسك الرئيسي**: **Task 4: Modeling KNN and Decision Trees with Hyperparameter Tuning**
* **المهام التقنية بالتفصيل**:
  1. بناء وضبط نموذج أقرب الجيران **K-Nearest Neighbors (KNN)** باستخدام خوارزمية البحث الشبكي `GridSearchCV` مع 5-Fold Stratified Cross-Validation.
  2. تجربة قيم $K \in \{3, 5, 7, 9, 11\}$ وأوزان المسافات والوصول إلى أفضل تركيبة ($K=7$ والمسافة الإقليدية بدقة 96.4%).
  3. بناء نموذج شجرة القرار **Decision Tree (DT)** وضبط معاملات التشذيب (Pruning) لمنع فرط التخصيص (Overfitting) مثل `max_depth=4` و `min_samples_leaf=2`.
  4. تصدير النماذج المدربة كملفات تشغيلية مستقلة في `models/knn.joblib` و `models/decision_tree.joblib`.
* **الملفات المسلمة من قبله**: كود تدريب وتيونينج KNN وشجرة القرار في `train_models_extensive.py` ونماذج `.joblib` المقابلة.
* **الشرائح التي يشرحها في البريزنتيشن**:
  * **Slide 9 (الجزء الأول)**: نظرية عمل KNN وتأثير المسافات، ونظرية عمل Decision Tree ومعايير نقاء جيني (Gini Impurity).
* **سؤال المناقشة المتوقع **: *"ليه الـ Decision Tree جابت أقل دقة (92.8%) مقارنة بباقي النماذج؟"*  
  **الإجابة النموذجية**: *"لأن شجرة القرار الفردية سريعة التأثر بالفروق الدقيقة في عينات التدريب وتميل لفرط التخصيص العالي، وحتى مع تحديد العمق (Pruning) تظل قدرتها التعميمية أضعف من النماذج التجميعية."*

---

### 5. إنجي (Engy) — مهندسة الشبكات العصبية والنماذج الشعاعية (Neural & Kernel Models Engineer)
* **نسبة المشاركة**: 16.67%
* **التاسك الرئيسي**: **Task 4: Modeling Artificial Neural Networks (ANN) & Support Vector Machines (SVM)**
* **المهام التقنية بالتفصيل**:
  1. تصميم وتدريب الشبكة العصبية الاصطناعية **Artificial Neural Network (MLPClassifier)** بمعمارية تحوي طبقة خفية (32 خلية عصبية)، ودالة تنشيط غير خطية `relu` ومعدل تعلم تكيفي.
  2. تطبيق تقنية تنظيم الأوزان L2 Regularization (`alpha=0.1`) في الشبكة العصبية لضمان عدم حدوث تشبع أو تذبذب في تدفق الانحدار.
  3. بناء نموذج آلة المتجهات الداعمة **Support Vector Machine (SVM)** بنواة شعاعية غير خطية `rbf` (Radial Basis Function Kernel).
  4. ضبط المعامل $C=0.1$ و $\gamma='scale'$ للحصول على أوسع هامش فاصل (Maximal Margin) وتفادي فرط التخصيص.
  5. تصدير النماذج في `models/artificial_neural_network_ann.joblib` و `models/support_vector_machine_svm.joblib`.
* **الملفات المسلمة من قبلها**: كود شبكة ANN ونموذج SVM وضبط معاملاتهما في `train_models_extensive.py` وملفات النماذج.
* **الشرائح التي تشرحها في البريزنتيشن**:
  * **Slide 9 (الجزء الثاني)**: شرح بنية الشبكة العصبية ANN، وكيفية استخدام كيرنل RBF في SVM لرفع الأبعاد وفصل البيانات غير الخطية.
* **سؤال المناقشة المتوقع **: *"ليه استخدمتم RBF Kernel في SVM مش Linear Kernel؟"*  
  **الإجابة النموذجية**: *"لأن الحدود الفاصلة بين الأنسجة الحميدة والخبيثة في الخلايا معقدة وغير خطية (Non-linear boundaries)، فنواة RBF بتنقل البيانات لفضاء رياضي أعلى بعداً يسهل فيه الفصل بدقة فائقة."*

---

### 6. كارين (Karen) — أخصائية النماذج التجميعية وتقييم المقاييس وتطبيق الويب (Ensemble, Benchmark & Web App Specialist)
* **نسبة المشاركة**: 16.67%
* **التاسك الرئيسي**: **Task 4 (Random Forest, Soft Voting) & Task 5 (Evaluation, Benchmark & Streamlit App)**
* **المهام التقنية بالتفصيل**:
  1. بناء نموذج الغابة العشوائية **Random Forest (RF)** عبر تجميع 100 شجرة قرار بطريقة الـ Bagging وضبط `max_depth=3` لمنع الـ Overfitting.
  2. بناء النموذج التجميعي الفائق **Soft Voting Meta-Ensemble** لدمج احتمالات أفضل ثلاثة نماذج (Random Forest + ANN + SVM).
  3. استخراج ورسم مصفوفات الارتباك الستة على عينة الاختبار ($N=139$) في رسم بياني موحد (`figures/7_all_confusion_matrices.png`).
  4. حساب وتفسير القيم السريرية: True Positives (46), True Negatives (89), False Positives (2), False Negatives (2).
  5. حساب ورسم منحنيات الـ ROC لجميع النماذج ومقارنة مساحات الـ AUC (`figures/8_roc_curves_comparison.png`).
  6. استخراج أهمية الخصائص الخلوية في Random Forest (`figures/10_rf_feature_importance.png`)، ورسم المقارنة الشاملة (`figures/9_metrics_comparison_bar.png`).
  7. تطوير واجهة تطبيق الويب التفاعلي بالكامل [`app.py`](file:///e:/advai/app.py) وتجهيز شاشة الاستقبال المخصصة لـ Eng Mathew، وإعداد نظام التحميل الفوري On-Demand.
* **الملفات المسلمة من قبلها**: `models/random_forest.joblib`، `models/ensemble_voting.joblib`، `app.py`، `figures/7, 8, 9, 10`، `model_comparison_results.csv`.
* **الشرائح التي تشرحها في البريزنتيشن**:
  * **Slide 10**: الشرح المفصل لمصفوفات الارتباك (Confusion Matrices) لكل الموديلات وكيف نجح الثلاثي في تفادي الـ False Negatives.
  * **Slide 11**: جدول المقارنة الشامل ومنحنيات الـ ROC-AUC وتفوق النماذج.
  * **Slide 12**: إعلان وتبرير اختيار Random Forest كبطل المشروع بفضل قابلية التفسير (Explainability).
  * **Slide 13**: عمل عرض حي (Live Demo) لتطبيق الويب، وخاتمة المشروع وآفاق التطوير المستقبلي.
* **سؤال المناقشة المتوقع **: *"طالما الـ ANN جابت أعلى ROC-AUC (0.9961)، ليه اخترتم Random Forest كنموذج فائز؟"*  
  **الإجابة النموذجية**: *"لأن في المجال الطبي الشفافية وتفسير القرار شرط أساسي؛ الـ ANN صندوق أسود لا يشرح قراره للطبيب، بينما Random Forest حقق نفس أعلى نسبة استرجاع ودقة (97.12% و 95.83%) ومعه تقرير Feature Importance الذي يثبت للطبيب سبب القرار."*

---

## 📈 جدول مقارنة نصيب كل عضو ومؤشرات التساوي (Equity Check)

| المعيار | ريمون | مارسلينو | بيشوى | ماريو | إنجي | كارين |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **مستوى الصعوبة** | متقدم (Architecture & CLI) | متوسط-متقدم (EDA 1) | متوسط-متقدم (EDA 2) | متقدم (Instance & Trees) | متقدم (Deep & Kernels) | متقدم (Ensembles, Metrics & App) |
| **عدد المخرجات البرمجية** | 4 ملفات تشغيل | 3 رسومات وكود | 3 رسومات وكود | 2 موديل وكود | 2 موديل وكود | 2 موديل، 4 رسومات وتطبيق ويب |
| **زمن العرض في البريزنتيشن** | 2.5 دقيقة | 2.5 دقيقة | 2.5 دقيقة | 2.5 دقيقة | 2.5 دقيقة | 2.5 دقيقة |
| **النسبة الإجمالية** | **16.67%** | **16.67%** | **16.67%** | **16.67%** | **16.67%** | **16.67%** |

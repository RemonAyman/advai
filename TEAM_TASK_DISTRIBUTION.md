# 👥 تقرير توزيع مهام المشروع بالتساوي على أعضاء الفريق (7 أشخاص)
## Project Task & Workload Distribution Matrix (Fair & Equal Division)

**اسم المشروع**: Supervised Machine Learning Classification Project — Breast Cancer Wisconsin (Original)  
**المستودع (Repository)**: [RemonAyman/advai](https://github.com/RemonAyman/advai)  
**عدد أعضاء الفريق**: 7 أعضاء  
**نسبة مساهمة كل عضو**: **14.28% لكل عضو بالتساوي وبنفس مستوى الصعوبة التقنية والجهد**

---

### 📊 نظرة عامة على مصفوفة التوزيع (Executive Workload Matrix)

| العضو (Member) | المسمى التقني في المشروع (Role) | التاسك المسؤول عنه (Assigned Task) | الملفات والكود الخاص به (Files & Code Owned) | شرائح العرض في البريزنتيشن (Slides) | نسبة الجهد |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **1. ريمون أيمن (Remon Ayman)** | **Team Lead & Preprocessing Architect** | **Task 2**: Data Preprocessing & Leakage Prevention | `run_project.py` (Prep), `imputer.joblib`, `scaler.joblib` | Slides 1, 2, 4 | **14.3%** |
| **2. مارسلينو (Marcelino)** | **Exploratory Data Analysis Specialist (Part 1)** | **Task 1** & **Task 3**: Dataset Exploration & Distributions | `figures/1, 2, 3`, Exploration Code | Slides 3, 5, 6 | **14.3%** |
| **3. بيشوى (Bishoy)** | **Advanced EDA & Cytology Relationships Specialist** | **Task 3**: Outliers, Correlation & Standardization | `figures/4, 5, 6`, Correlation Analysis | Slides 6, 7, 8 | **14.3%** |
| **4. ماريو (Mario)** | **Supervised ML Engineer (Instance & Tree Models)** | **Task 4**: KNN & Decision Tree Modeling & Tuning | `models/knn.joblib`, `models/decision_tree.joblib` | Slide 9 (Part 1) | **14.3%** |
| **5. إنجي (Engy)** | **Supervised ML Engineer (Neural & Kernel Models)** | **Task 4**: ANN (Neural Networks) & SVM Modeling & Tuning | `models/ann.joblib`, `models/svm.joblib` | Slide 9 (Part 2) | **14.3%** |
| **6. كارين (Karen)** | **Ensemble Learning & Clinical Evaluation Specialist** | **Task 4 & 5**: Random Forest, Voting Ensemble & Confusion Matrices | `models/rf.joblib`, `models/ensemble_voting.joblib`, `figures/7` | Slides 9, 10 | **14.3%** |
| **7. ديفيد (David)** | **Benchmark, Deployment & Software Engineer** | **Task 5 & Deployment**: ROC-AUC, Web App & CLI Tool | `app.py`, `predict_cli.py`, `run_app.bat`, `figures/8, 10` | Slides 11, 12, 13 | **14.3%** |

---

## 🔍 التفاصيل الدقيقة لمسؤولية كل عضو (Detailed Breakdown)

---

### 1. ريمون أيمن (Remon Ayman) — مهندس معالجة البيانات ومنع التسرب (Team Lead & Preprocessing Architect)
* **نسبة المشاركة**: 14.3%
* **التاسك الرئيسي**: **Task 2: Data Preprocessing & Data Leakage Prevention**
* **المهام التقنية بالتفصيل**:
  1. فحص القيم المفقودة مشفرة كعلامة استفهام `'?'` في متغير `Bare_Nuclei` واستبدالها بـ `NaN`.
  2. تطبيق استراتيجية التعويض بالوسيط (`SimpleImputer(strategy='median')`) والتأكد بنسبة 100% أن التدريب يتم على `X_train` فقط لمنع الـ Data Leakage.
  3. رصد وحذف الـ 8 صفوف المكررة بالكامل (نفس المعرف ونفس الخصائص) للحفاظ على استقلالية العينات.
  4. حذف المعرف الفريد `Sample_Code_Number` لمنع النموذج من حفظ أرقام العينات عشوائياً.
  5. تقسيم البيانات بنسبة 80% تدريب (552 عينة) و 20% اختبار (139 عينة) بنظام التوزيع الطبقي (`stratify=y`).
  6. تدريب وتطبيق `StandardScaler` على بيانات التدريب ثم تحويل بيانات الاختبار بنفس المعايير ($\mu, \sigma$).
* **الملفات المسلمة من قبله**: مسار Preprocessing في `run_project.py`، `models/imputer.joblib`، `models/scaler.joblib`.
* **الشرائح التي يشرحها في البريزنتيشن**:
  * **Slide 1**: عنوان المشروع والمقدمة.
  * **Slide 2**: المشكلة السريرية (Problem Statement) والأهداف الأساسية.
  * **Slide 4**: خطوات تنظيف البيانات بالتفصيل وشرح كيفية منع تسرب البيانات (Data Leakage).
* **سؤال المناقشة المتوقع **: *"إزاي تأكدت إن الـ StandardScaler والـ Imputer معملوش تسريب داتا من الـ Test للـ Train؟"*  
  **الإجابة النموذجية**: *"عملنا fit على X_train فقط، واحتفظنا ببيانات الـ Test معزولة تماماً وطبقنا عليها transform فقط بالقيم اللي اتعلمها من التدريب."*

---

### 2. مارسلينو (Marcelino) — أخصائي استكشاف البيانات والتوزيعات (EDA Specialist - Part 1)
* **نسبة المشاركة**: 14.3%
* **التاسك الرئيسي**: **Task 1: Dataset Exploration** & **Task 3: EDA (Missingness, Target & Histograms)**
* **المهام التقنية بالتفصيل**:
  1. قراءة وتحميل الداتا عبر Pandas وفحص الأبعاد (699 صف × 11 عمود) وأنواع البيانات.
  2. توثيق معاني الخصائص الخلوية التسعة وتحديد المتغير الهدف (`Class`).
  3. إنشاء وتحليل رسم القيم المفقودة (`figures/1_missing_values.png`) وتوضيح أن نسبة الفقد 2.3% فقط في متغير الأنوية العارية.
  4. إنشاء وتحليل رسم التوزيع الفئوي (`figures/2_target_distribution.png`) وتوضيح أن 65.6% حميد و 34.4% خبيث.
  5. إنشاء شبكة المدرجات التكرارية 3×3 لكل الخصائص الخلوية التسعة (`figures/3_feature_histograms.png`) وتفسير الانحراف الإيجابي الحاد (Right-Skewness).
* **الملفات المسلمة من قبله**: كود استكشاف البيانات، `figures/1_missing_values.png`، `figures/2_target_distribution.png`، `figures/3_feature_histograms.png`.
* **الشرائح التي يشرحها في البريزنتيشن**:
  * **Slide 3**: مصدر البيانات، أسماء الخصائص ومعانيها الطبية ودرجاتها من 1 إلى 10.
  * **Slide 5**: تحليل القيم المفقودة ونسبة التوزيع الفئوي بين الورم الحميد والخبيث.
  * **Slide 6 (الجزء الأول)**: تفسير الـ Histograms ولماذا تتكدس الخلايا السليمة عند الدرجة 1.
* **سؤال المناقشة المتوقع **: *"ليه كل المدرجات التكرارية (Histograms) مايلة لليمين (Right-skewed)؟"*  
  **الإجابة النموذجية**: *"لأن أغلب العينات في الطبيعة عبارة عن خلايا طبيعية غير سرطانية بتاخد درجات منخفضة (1)، بينما الدرجات العالية من 3 لـ 10 بتمثل حالات نادرة للتحول السرطاني."*

---

### 3. بيشوى (Bishoy) — أخصائي العلاقات الخلوية والارتباط (EDA Specialist - Part 2)
* **نسبة المشاركة**: 14.3%
* **التاسك الرئيسي**: **Task 3: Advanced EDA (Outliers, Correlation & Standardization Impact)**
* **المهام التقنية بالتفصيل**:
  1. إنشاء وتحليل المخططات الصندوقية (`figures/4_feature_boxplots.png`) لرصد القيم المتطرفة (Outliers).
  2. تقديم التبرير الطبي لعدم حذف القيم المتطرفة إحصائياً في `Mitoses` لأنها علامات سريرية حقيقية للورم الخبيث وليست أخطاء داتا.
  3. إنشاء وتحليل خريطة الارتباط الحرارية (`figures/5_correlation_heatmap.png`) وكشف ظاهرة التعدد الخطي (Multicollinearity).
  4. دراسة العلاقة بين تجانس حجم الخلية وشكلها ($r = 0.91$) وارتباطهما بالورم ($r = 0.82$).
  5. إنشاء رسم مقارنة توزيع الخصائص قبل وبعد التوحيد القياسي (`figures/6_before_after_standardization.png`).
* **الملفات المسلمة من قبله**: كود الارتباط والـ Boxplots، `figures/4_feature_boxplots.png`، `figures/5_correlation_heatmap.png`، `figures/6_before_after_standardization.png`.
* **الشرائح التي يشرحها في البريزنتيشن**:
  * **Slide 6 (الجزء الثاني)**: تحليل الـ Boxplots والقيم المتطرفة وتبرير عدم حذفها.
  * **Slide 7**: خريطة الارتباط الحرارية وظاهرة الـ Multicollinearity بين حجم وشكل الخلية.
  * **Slide 8**: مقارنة توزيع البيانات قبل وبعد الـ Standardization ولماذا هي ضرورية للخوارزميات.
* **سؤال المناقشة المتوقع **: *"ليه محذفتش الـ Outliers اللي ظاهرة في Boxplot بتاع الـ Mitoses؟"*  
  **الإجابة النموذجية**: *"لأن في علم الأورام الانقسام السريع للخلية هو أهم إشارة لوجود ورم خبيث عنيف، وحذف القيم دي يعتبر إتلاف لإشارات تشخيصية حيوية، فالحل الأمثل كان توحيد مقياسها مش حذفها."*

---

### 4. ماريو (Mario) — مهندس نماذج التعلم الآلي القائمة على المسافات والأشجار (ML Engineer - KNN & Decision Tree)
* **نسبة المشاركة**: 14.3%
* **التاسك الرئيسي**: **Task 4: Model Implementation (KNN & Decision Tree)**
* **المهام التقنية بالتفصيل**:
  1. بناء خوارزمية **K-Nearest Neighbors (KNN)** وضبط المعاملات الفائقة عبر `GridSearchCV`:
     * البحث في عدد الجيران: `n_neighbors = [3, 5, 7, 9, 11]`.
     * فحص أوزان التصويت: `weights = ['uniform', 'distance']`.
     * مقارنة مقاييس المسافة: `metric = ['euclidean', 'manhattan']`.
  2. بناء خوارزمية **Decision Tree (DT)** وضبط معاملاتها:
     * فحص معيار التقسيم: `criterion = ['gini', 'entropy']`.
     * ضبط أقصى عمق: `max_depth = [3, 4, 5, 6, None]`.
     * فحص شروط التقسيم والأوراق: `min_samples_split`, `min_samples_leaf`.
  3. تنفيذ التحقق المتقاطع الطبقي المكون من 5 طيات (**5-Fold Stratified CV**) واختيار أفضل النماذج.
* **الملفات المسلمة من قبله**: كود تدريب وضبط KNN و DT، وحفظ الموديلات في `models/knn.joblib` و `models/decision_tree.joblib`.
* **الشرائح التي يشرحها في البريزنتيشن**:
  * **Slide 9 (الجزء الأول)**: شرح فكرة عمل KNN وكيفية اختيار الجيران الأقرب (Optimal: $k=7$)، وفكرة شجرة القرار والحدود المتعامدة (Optimal: depth=4).
* **سؤال المناقشة المتوقع **: *"ليه شجرة القرار المنفردة (Decision Tree) جابت أقل دقة واسترجاع (89.58% Recall)؟"*  
  **الإجابة النموذجية**: *"لأن شجرة القرار المنفردة بتعاني من تباين عالي (High Variance) وحساسية شديدة لأي تغير في الداتا، وبتقسم الحدود بخطوط عمودية حادة فقط، وعشان كده بنحتاج ندمجها في Random Forest."*

---

### 5. إنجي (Engy) — مهندسة الشبكات العصبية والنماذج الهامشية (ML Engineer - ANN & SVM)
* **نسبة المشاركة**: 14.3%
* **التاسك الرئيسي**: **Task 4: Model Implementation (ANN & Support Vector Machine)**
* **المهام التقنية بالتفصيل**:
  1. بناء وضبط الشبكة العصبية الاصطناعية **Artificial Neural Network (MLPClassifier)**:
     * تدريب الشبكة لـ 2,500 دورة تدريبية (Max Iterations) للوصول للتقارب الكامل.
     * تجربة هياكل الطبقات الخفية: `hidden_layer_sizes = [(32,), (64,), (64, 32)]`.
     * تجربة دوال التنشيط غير الخطية: `activation = ['relu', 'tanh']`.
     * ضبط معامل التنظيم L2 (Weight Decay) ومعدل التعلم: `alpha`, `learning_rate_init`.
  2. بناء وضبط آلة المتجهات الداعمة **Support Vector Machine (SVM)**:
     * تجربة الأنوية: Linear و Radial Basis Function (RBF Kernel).
     * ضبط معامل العقوبة $C$ وضبط معامل الانتشار $\gamma$ (Gamma).
  3. تطبيق 5-Fold Stratified Cross-Validation لتحقيق التوازن بين Bias و Variance.
* **الملفات المسلمة من قبلها**: كود تدريب وضبط ANN و SVM، وحفظ الموديلات في `models/artificial_neural_network_ann.joblib` و `models/support_vector_machine_svm.joblib`.
* **الشرائح التي تشرحها في البريزنتيشن**:
  * **Slide 9 (الجزء الثاني)**: شرح بنية الشبكة العصبية متعددة الطبقات ودور دالة ReLU، وشرح فكرة الفصل بأقصى هامش ونواة RBF في SVM.
* **سؤال المناقشة المتوقع للمشرف**: *"إيه دور الـ RBF Kernel في خوارزمية الـ SVM؟"*  
  **الإجابة النموذجية**: *"الـ RBF Kernel بتنقل البيانات لفضاء رياضي لا نهائي الأبعاد (Infinite Dimensional Space) عشان تقدر تفصل بين الحالات اللي مستحيل فصلها بخط مستقيم في الفضاء الأصلي."*

---

### 6. كارين (Karen) — أخصائية النماذج التجميعية ومصفوفات الارتباك (Ensemble Learning & Confusion Matrix Specialist)
* **نسبة المشاركة**: 14.3%
* **التاسك الرئيسي**: **Task 4 (Random Forest & Voting Ensemble)** & **Task 5: Confusion Matrices & Medical Evaluation**
* **المهام التقنية بالتفصيل**:
  1. بناء نموذج الغابة العشوائية **Random Forest (RF)** عبر تجميع 100 إلى 300 شجرة قرار بطريقة الـ Bagging وتقليل الارتباط بين الأشجار (Feature Subspacing).
  2. بناء النموذج التجميعي الفائق **Soft Voting Meta-Ensemble** لدمج احتمالات أفضل ثلاثة نماذج (Random Forest + ANN + SVM).
  3. استخراج ورسم مصفوفات الارتباك الستة على عينة الاختبار ($N=139$) في رسم بياني موحد (`figures/7_all_confusion_matrices.png`).
  4. حساب وتفسير القيم السريرية: True Positives (46), True Negatives (89), False Positives (2), False Negatives (2).
  5. شرح لماذا يعتبر الـ False Negative أخطر خطأ طبي في تشخيص الأورام.
* **الملفات المسلمة من قبلها**: كود تدريب RF والـ Ensemble، وحفظ `models/random_forest.joblib` و `models/ensemble_voting.joblib`، ورسم `figures/7_all_confusion_matrices.png`.
* **الشرائح التي تشرحها في البريزنتيشن**:
  * **Slide 9 (الجزء التجميعي)**: شرح طريقة الـ Bagging في Random Forest وطريقة الـ Soft Voting Ensemble.
  * **Slide 10**: الشرح المفصل لمصفوفات الارتباك (Confusion Matrices) لكل الموديلات وكيف نجح الثلاثي في تفادي الـ False Negatives.
* **سؤال المناقشة المتوقع **: *"ليه بنفضل الـ Recall على الـ Precision في أمراض السرطان؟"*  
  **الإجابة النموذجية**: *"لأن الـ False Negative معناه مريض عنده سرطان روح بيته على إنه سليم، وده هيموت، بينما الـ False Positive معناه مريض سليم هيعمل فحص تأكيدي إضافي، فالأولى حياة والتانية إزعاج."*

---

### 7. ديفيد (David) — مهندس المقارنات وتطوير وبرمجة التطبيقات (Benchmark, Deployment & App Engineer)
* **نسبة المشاركة**: 14.3%
* **التاسك الرئيسي**: **Task 5 (ROC-AUC, Model Justification) & Interactive App Deployment**
* **المهام التقنية بالتفصيل**:
  1. حساب ورسم منحنيات الـ ROC لجميع النماذج ومقارنة مساحات الـ AUC (`figures/8_roc_curves_comparison.png`).
  2. استخراج درجات أهمية الخصائص الخلوية عبر مقياس Gini في Random Forest (`figures/10_rf_feature_importance.png`).
  3. تجميع جدول المقارنة الشامل (`model_comparison_results.csv`) وصياغة المبررات العلمية لاختيار Random Forest كنموذج فائز للعيادات الطبية.
  4. برمجة تطبيق الويب التفاعلي الكامل بلغة Streamlit في ملف [`app.py`](file:///E:/advai/app.py) وتوفير أزرار الحالات الجاهزة وسلايدرز الفحص والرسومات الحية.
  5. برمجة أداة الفحص عبر سطر الأوامر [`predict_cli.py`](file:///E:/advai/predict_cli.py) ومشغل الويندوز التلقائي بنقرة واحدة [`run_app.bat`](file:///E:/advai/run_app.bat).
* **الملفات المسلمة من قبله**: `app.py`، `predict_cli.py`، `run_app.bat`، `figures/8_roc_curves_comparison.png`، `figures/10_rf_feature_importance.png`، `model_comparison_results.csv`.
* **الشرائح التي يشرحها في البريزنتيشن**:
  * **Slide 11**: جدول المقارنة الشامل ومنحنيات الـ ROC-AUC وتفوق النماذج.
  * **Slide 12**: إعلان وتبرير اختيار Random Forest كبطل المشروع بفضل قابلية التفسير (Explainability).
  * **Slide 13**: عمل عرض حي (Live Demo) لتطبيق الويب، وخاتمة المشروع وآفاق التطوير المستقبلي.
* **سؤال المناقشة المتوقع **: *"طالما الـ ANN جابت أعلى ROC-AUC (0.9961)، ليه اخترتم Random Forest كنموذج فائز؟"*  
  **الإجابة النموذجية**: *"لأن في المجال الطبي الشفافية وتفسير القرار شرط أساسي للموافقة السريرية؛ الـ ANN صندوق أسود، بينما Random Forest حقق نفس أعلى نسبة استرجاع ودقة (97.12% و 95.83%) ومعاها أهمية الخصائص اللي بتثبت للطبيب سبب القرار."*

---

## 📈 جدول مقارنة نصيب كل عضو ومؤشرات التساوي (Equity Check)

| المعيار | ريمون | مارسلينو | بيشوى | ماريو | إنجي | كارين | ديفيد |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **مستوى الصعوبة** | متقدم (Architecture) | متوسط-متقدم (EDA) | متوسط-متقدم (EDA) | متقدم (Algorithms) | متقدم (Deep & Kernels) | متقدم (Ensembles & Metrics) | متقدم (App & Benchmark) |
| **عدد المخرجات البرمجية** | 3 ملفات | 3 رسومات وكود | 3 رسومات وكود | 2 موديل وكود | 2 موديل وكود | 2 موديل ورسمة | 3 تطبيقات ورسمتين |
| **زمن العرض في البريزنتيشن** | 2 دقيقة | 2 دقيقة | 2 دقيقة | 2 دقيقة | 2 دقيقة | 2 دقيقة | 2 دقيقة |
| **النسبة الإجمالية** | **14.3%** | **14.3%** | **14.3%** | **14.3%** | **14.3%** | **14.3%** | **14.3%** |


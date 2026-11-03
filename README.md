# 🎗️ Breast Cancer Wisconsin (Original) Supervised Classification System
### Complete Machine Learning Pipeline, Clinical Evaluation & Web Diagnostic Application

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Repository](https://img.shields.io/badge/GitHub-RemonAyman%2Fadvai-181717.svg)](https://github.com/RemonAyman/advai)
[![Project Status](https://img.shields.io/badge/All%20Tasks-100%25%20Completed-brightgreen.svg)]()

---

## 📌 1. Project Overview & Clinical Context
This project presents an end-to-end supervised machine learning classification system developed on the landmark **Breast Cancer Wisconsin (Original)** dataset from the **UCI Machine Learning Repository**, originally gathered by **Dr. William H. Wolberg** at the University of Wisconsin Hospitals, Madison.

The objective is to accurately and reliably classify breast tumor Fine Needle Aspirate (FNA) biopsies as **Benign (حميد)** or **Malignant (خبيث)**. In oncological diagnostics, **False Negatives are fatal**, as missing an active malignancy delays life-saving interventions. This system implements a leak-free machine learning workflow, evaluates five supervised algorithms, trains a soft-voting ensemble, and provides an interactive clinical web application for second-opinion diagnosis.

---

## 🗂️ 2. Repository Structure

```text
advai/
│
├── breast+cancer+wisconsin+original/   # Official UCI raw dataset & documentation
│   ├── breast-cancer-wisconsin.data    # 699 clinical instances (CSV format)
│   ├── breast-cancer-wisconsin.names   # Official dataset documentation & attribute descriptions
│   └── ...
│
├── figures/                            # High-resolution (300 DPI) publication-quality charts
│   ├── 1_missing_values.png            # Missing data breakdown per feature
│   ├── 2_target_distribution.png       # Benign vs Malignant class distribution
│   ├── 3_feature_histograms.png        # 3x3 grid of cytological feature histograms
│   ├── 4_feature_boxplots.png          # Boxplots for outlier & range analysis
│   ├── 5_correlation_heatmap.png       # Correlation matrix with multicollinearity
│   ├── 6_before_after_standardization.png # Feature distribution scaling comparison
│   ├── 7_all_confusion_matrices.png    # 5-Model confusion matrix grid
│   ├── 8_roc_curves_comparison.png     # ROC curves & AUC comparison
│   ├── 9_metrics_comparison_bar.png    # Bar chart of Accuracy, Precision, Recall, F1
│   └── 10_rf_feature_importance.png    # Random Forest Gini feature importances
│
├── models/                             # Pre-trained models & preprocessing artifacts
│   ├── imputer.joblib                  # Learned training median imputer
│   ├── scaler.joblib                   # Learned training StandardScaler
│   ├── random_forest.joblib            # Champion Random Forest model
│   ├── support_vector_machine_svm.joblib # Tuned RBF Support Vector Machine
│   ├── artificial_neural_network_ann.joblib # Tuned Multi-Layer Perceptron (ANN)
│   ├── knn.joblib                      # Tuned K-Nearest Neighbors
│   ├── decision_tree.joblib            # Tuned Decision Tree
│   ├── ensemble_voting.joblib          # Soft Voting Meta-Ensemble (RF + ANN + SVM)
│   └── metadata.json                   # Metadata & feature definitions for web app
│
├── app.py                              # Interactive Streamlit Web Application
├── predict_cli.py                      # Interactive Command-Line Prediction Tool
├── run_app.bat                         # One-click Windows desktop launcher for Web App
├── run_project.py                      # Standalone complete pipeline execution script
├── train_models_extensive.py           # Deep & extensive model training script
├── build_and_run_notebook.py           # Programmatic notebook builder & executor
├── generate_presentation.py            # Automated PowerPoint (.pptx) generator
├── breast_cancer_classification.ipynb  # Executed Jupyter Notebook with outputs & plots
├── breast_cancer_classification_presentation.pptx # 13-slide PowerPoint Presentation
├── presentation_slides_guide.md        # Comprehensive slide-by-slide script & talking notes
├── model_comparison_results.csv        # Unified evaluation metrics comparison table
├── requirements.txt                    # Project dependency specifications
├── .gitignore                          # Git ignore definitions
└── README.md                           # Master documentation
```

---

## 🔬 3. Detailed Walkthrough of All Required Tasks

### Task 1: Dataset Exploration
* **Requirements**: Load dataset using Pandas, examine structure, dimensions, data types, identify target and predictive features, review documentation, and analyze descriptive statistics, missing values, duplicates, and class balance.
* **Implementation & Findings**:
  * **Dimensions**: 699 rows and 11 attributes.
  * **Identifier Attribute**: `Sample_Code_Number` (arbitrary patient ID).
  * **9 Predictive Features** (integer scale $1$ to $10$, where $1$ is normal healthy morphology and $10$ is severe abnormality):
    1. `Clump_Thickness`: Agglomeration of cells in clusters.
    2. `Uniformity_of_Cell_Size`: Uniformity in cell dimensions.
    3. `Uniformity_of_Cell_Shape`: Geometric cell boundary regularity.
    4. `Marginal_Adhesion`: Intercellular cohesion (loss enables metastasis).
    5. `Single_Epithelial_Cell_Size`: Epithelial cell cytoplasmic volume.
    6. `Bare_Nuclei`: Nuclei stripped of cytoplasm (key malignancy marker).
    7. `Bland_Chromatin`: Nuclear chromatin texture and granularity.
    8. `Normal_Nucleoli`: Prominence and hyperactivity of nucleoli.
    9. `Mitoses`: Rate of mitotic cellular divisions.
  * **Target Attribute**: `Class` (`2` = Benign, `4` = Malignant).
  * **Missing Values**: Exactly 16 occurrences of `'?'` in `Bare_Nuclei` (2.3% of records). All other columns have 0 missing values.
  * **Duplicate Records**: 8 exact duplicate rows (identical ID and identical features).
  * **Class Distribution**: 458 Benign (65.5%) vs. 241 Malignant (34.5%).

---

### Task 2: Data Preprocessing & Data Leakage Prevention
* **Requirements**: Handle missing values, resolve duplicates, remove irrelevant identifiers, split 80:20 with stratification, apply standardization, and guarantee zero data leakage.
* **Implementation & Strategy**:
  1. **Missing Value Imputation**: Replaced `'?'` with `NaN`. Missing values in `Bare_Nuclei` imputed using **Median Imputation** (`median = 1.0`). To prevent **Data Leakage**, the `SimpleImputer` was fitted **exclusively on the training set** (`X_train`) and then applied to transform both `X_train` and `X_test`.
  2. **Deduplication**: Removed the 8 exact duplicate rows to maintain true statistical independence, leaving **691 clean patient records**.
  3. **Identifier Removal**: Dropped `Sample_Code_Number` so models cannot learn spurious memorization artifacts.
  4. **Target Encoding**: Mapped `Benign (2) -> 0` and `Malignant (4) -> 1` so that **Malignant is treated strictly as the Positive Class**.
  5. **Stratified 80:20 Partition**:
     * **Training Set**: 552 samples (80%) — 362 Benign, 190 Malignant (34.4% positive).
     * **Testing Set**: 139 samples (20%) — 91 Benign, 48 Malignant (34.5% positive).
  6. **Feature Standardization**: `StandardScaler` was fitted **strictly on `X_train` only**, computing training mean ($\mu$) and standard deviation ($\sigma$). These parameters were then used to standardize both `X_train` and `X_test` without contamination.

---

### Task 3: Exploratory Data Analysis (EDA)
Six comprehensive visualization figures were generated, each equipped with descriptive titles, labeled axes, and clinical interpretations:

| Figure File | Analysis Type | Key Statistical & Clinical Interpretation |
| :--- | :--- | :--- |
| `1_missing_values.png` | **Missing Values Analysis** | Missing data is confined strictly to `Bare_Nuclei` (16 cases / 2.3%). Median imputation preserves 100% of samples without distorting feature distribution. |
| `2_target_distribution.png` | **Target Class Countplot** | Demonstrates class balance (65.6% Benign vs 34.4% Malignant in clean data). Well-suited for supervised learning with stratified sampling. |
| `3_feature_histograms.png` | **Feature Distributions** | Marked positive (right) skewness across all 9 features. Healthy cells heavily cluster at score 1; scores $\ge 3$ represent progressively severe malignant transformations. |
| `4_feature_boxplots.png` | **Outlier Analysis** | Extreme values in `Mitoses` and `Marginal_Adhesion` appear as statistical outliers, but represent **true pathological malignancy markers**; retaining them is clinically essential. |
| `5_correlation_heatmap.png` | **Multicollinearity & Relationships** | Extreme correlation ($r = 0.91$) between `Uniformity_of_Cell_Size` and `Uniformity_of_Cell_Shape`. Both features, along with `Bare_Nuclei`, correlate highest with cancer ($r = 0.82$). |
| `6_before_after_standardization.png` | **Standardization Impact** | Confirms shift to zero mean and unit variance without altering distribution shapes, eliminating distance scale bias for KNN, SVM, and ANN. |

---

### Task 4: Model Implementation & Hyperparameter Tuning
Five diverse supervised machine learning architectures were implemented and extensively tuned using **5-Fold Stratified Cross-Validation (`GridSearchCV`)** optimizing for **Recall (Sensitivity)**:

1. **K-Nearest Neighbors (KNN)**: Non-parametric instance-based learner.
   * *Optimal Parameters*: `n_neighbors = 7`, `metric = 'euclidean'`, `weights = 'uniform'`
2. **Decision Tree (DT)**: Recursive axis-aligned binary partitioner.
   * *Optimal Parameters*: `criterion = 'gini'`, `max_depth = 4`, `min_samples_split = 2`, `min_samples_leaf = 2`
3. **Random Forest (RF)**: Ensemble of de-correlated bagged decision trees with feature subspacing.
   * *Optimal Parameters*: `n_estimators = 100`, `max_depth = 3`, `min_samples_split = 10`, `criterion = 'gini'`
4. **Artificial Neural Network (ANN / MLP)**: Multi-Layer Perceptron trained with backpropagation and L2 regularization.
   * *Optimal Parameters*: `hidden_layer_sizes = (32,)`, `activation = 'relu'`, `alpha = 0.1`, `learning_rate_init = 0.001`, `max_iter = 2500`
5. **Support Vector Machine (SVM)**: Maximum-margin hyperplane with non-linear Radial Basis Function (RBF) kernel.
   * *Optimal Parameters*: `kernel = 'rbf'`, `C = 0.1`, `gamma = 'scale'`
6. **Soft-Voting Meta-Ensemble**: Blends predicted class probability distributions from Random Forest, ANN, and SVM.

---

### Task 5: Model Evaluation & Benchmark Comparison
All models were evaluated on the held-out unseen test set ($N=139$ patients: 91 Benign, 48 Malignant):

#### 📊 Master Model Performance Comparison Table:

| Classification Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Artificial Neural Network (ANN)** | **97.12%** | **95.83%** | **95.83%** | **0.9583** | **0.9961** |
| **Support Vector Machine (SVM)** | **97.12%** | **95.83%** | **95.83%** | **0.9583** | **0.9945** |
| **Voting Ensemble (RF+ANN+SVM)** | **97.12%** | **95.83%** | **95.83%** | **0.9583** | **0.9950** |
| **Random Forest (RF) ⭐** | **97.12%** | **95.83%** | **95.83%** | **0.9583** | **0.9929** |
| **K-Nearest Neighbors (KNN)** | 96.40% | 97.78% | 91.67% | 0.9462 | 0.9823 |
| **Decision Tree (DT)** | 92.81% | 89.58% | 89.58% | 0.8958 | 0.9755 |

#### 🎯 Confusion Matrix Insights ($N=139$):
* **Random Forest, ANN, SVM, and Ensemble**:
  * **True Benign (TN)**: 89 patients
  * **False Malignant (FP)**: 2 patients (unnecessary confirmatory biopsy, non-fatal)
  * **False Benign (FN)**: **Only 2 patients missed out of 48**
  * **True Malignant (TP)**: 46 patients correctly diagnosed
  * **Recall = 95.83%** | **Accuracy = 97.12%**
* **KNN**: Missed 4 malignant cases (Recall = 91.67%).
* **Decision Tree**: Missed 5 malignant cases and 5 benign cases (Recall = 89.58%, Accuracy = 92.81%).

#### 🏆 Best-Performing Model Selection & Clinical Justification:
While ANN achieved the marginally highest ROC-AUC ($0.9961$), **Random Forest (RF)** is crowned as the **Champion Model for Clinical Deployment**:
1. **Tied for Peak Clinical Performance**: Matches ANN and SVM with 97.12% Accuracy and 95.83% Recall.
2. **Transparent Interpretability**: Random Forest provides exact Gini feature importances, demonstrating that `Uniformity_of_Cell_Size`, `Bare_Nuclei`, and `Uniformity_of_Cell_Shape` are the primary diagnostic drivers, aligning with peer-reviewed pathology literature.
3. **Overfitting Resistance**: Bagging 100 de-correlated trees stabilizes predictions against biopsy noise and physician score variance.
4. **Computational Stability**: Instant execution without gradient vanishing, local minima, or convergence instability.

---

### Task 6: Presentation Slides
* **Delivered Artifact**: `breast_cancer_classification_presentation.pptx`
* **Features**:
  * 13 professionally styled, widescreen (16:9) slides.
  * Modern medical tech palette (Deep Navy `#1A365D`, Medical Teal `#0E7490`, Rose Red `#E11D48`).
  * Embedded high-resolution figures for missing data, class distribution, histograms, boxplots, correlation heatmap, standardization, confusion matrices, ROC curves, and feature importance.
  * Formatted tables and structured bullet points.
* **Presenter Guide**: Complete slide-by-slide script, timing, and speaking notes are documented in `presentation_slides_guide.md`.

---

## 🚀 4. How to Run the Project & Use the Applications

### Step 1: Install Dependencies
Open PowerShell or your command prompt in `E:\advai` and run:
```bash
pip install -r requirements.txt
```

---

### Step 2: Run the Interactive Web Application (Streamlit)
You can launch the web application in two ways:

#### Option A: One-Click Windows Launcher (Easiest)
Simply double-click the file:
```text
run_app.bat
```

#### Option B: Terminal Command
```bash
streamlit run app.py
```
This automatically opens your browser at `http://localhost:8501`.

#### 📱 How to Use the Web Application:
1. **Select Model**: In the left sidebar, choose your model (default: *Random Forest (Recommended)* or *Voting Ensemble*).
2. **Quick Presets**: Click on *"Typical Benign Patient"*, *"Typical Malignant Patient"*, or *"Borderline / Equivocal Case"* to auto-populate test values.
3. **Manual Adjustments**: Adjust any of the 9 cytological sliders (1 to 10) according to FNA biopsy findings.
4. **Predict**: Click **"🔬 Analyze Biopsy & Predict Diagnosis"**.
5. **View Results**:
   * Green badge: **BENIGN (ورم حميد)** with confidence score (e.g. 99.1% Benign).
   * Red alert badge: **MALIGNANT (ورم خبيث)** with malignancy probability and clinical recommendation.
   * View the horizontal bar chart comparing the patient's score against normal healthy baseline.
6. **Benchmark Tab**: View cross-model comparison tables, confusion matrices, and ROC curves directly in the app.

---

### Step 3: Run the Command-Line Interface (CLI Tool)
For rapid testing directly in your terminal:
```bash
python predict_cli.py
```
* Enter scores from 1 to 10 for each cytological feature or press **Enter** to accept default values.
* The terminal will immediately output the diagnosis and probability.

---

### Step 4: Open and Run the Jupyter Notebook
The notebook is fully executed and pre-rendered with all outputs and figures:
```bash
jupyter notebook breast_cancer_classification.ipynb
```
You can also open `breast_cancer_classification.ipynb` directly in **VS Code**, **PyCharm**, or **Google Colab**.

---

### Step 5: Re-train Models or Re-generate Slides (Optional)
* To run the extensive training pipeline and update all model `.joblib` files:
  ```bash
  python train_models_extensive.py
  ```
* To regenerate the PowerPoint presentation:
  ```bash
  python generate_presentation.py
  ```

---

## 👥 5. Team Workload & Task Distribution (تقسيم المهام على الفريق)
تم توزيع جميع مهام المشروع التقنية والتحليلية بالتساوي التام على أعضاء الفريق السبعة (**14.28% لكل عضو** مع التكافؤ التام في الجهد التقني ومخرجات الكود):

| العضو (Member) | الدور التقني (Role) | التاسك المسؤول عنه (Task) | الملفات والكود الخاص به (Deliverables) | شرائح البريزنتيشن (Slides) |
| :--- | :--- | :--- | :--- | :---: |
| **1. ريمون أيمن (Remon Ayman)** | **Team Lead & Preprocessing Architect** | **Task 2**: تنظيف البيانات ومنع تسرب البيانات (Data Leakage) | مسار المعالجة في `run_project.py`، `imputer.joblib`، `scaler.joblib` | Slides 1, 2, 4 |
| **2. مارسلينو (Marcelino)** | **Exploration & EDA Specialist (Part 1)** | **Task 1 & Task 3**: استكشاف البيانات ورسوم التوزيعات | كود الاستكشاف، `figures/1, 2, 3` (Missing, Target, Histograms) | Slides 3, 5, 6 |
| **3. بيشوى (Bishoy)** | **Advanced EDA & Relationships Specialist** | **Task 3**: تحليل الشواذ والارتباط وتوحيد المقياس | كود الارتباط، `figures/4, 5, 6` (Boxplots, Heatmap, Standardization) | Slides 6, 7, 8 |
| **4. ماريو (Mario)** | **Supervised ML Engineer (KNN & Tree)** | **Task 4**: خوارزميات KNN وشجرة القرار والضبط | كود التدريب، `models/knn.joblib`، `models/decision_tree.joblib` | Slide 9 (Part 1) |
| **5. إنجي (Engy)** | **Supervised ML Engineer (ANN & SVM)** | **Task 4**: الشبكات العصبية العميقة و SVM | كود التدريب، `models/ann.joblib`، `models/svm.joblib` | Slide 9 (Part 2) |
| **6. كارين (Karen)** | **Ensemble Learning & Clinical Evaluation** | **Task 4 & 5**: الغابات العشوائية ومصفوفات الارتباك | كود RF والـ Ensemble، `models/random_forest.joblib`، `figures/7` | Slides 9, 10 |
| **7. ديفيد (David)** | **Deployment, Benchmark & App Engineer** | **Task 5 & App**: منحنيات ROC، تطبيق الويب وسطر الأوامر | تطبيق الويب `app.py`، أداة `predict_cli.py`، `run_app.bat`، `figures/8, 10` | Slides 11, 12, 13 |

> للمزيد من التفاصيل الدقيقة حول أسئلة المناقشة المتوقعة لكل عضو وإجاباتها النموذجية، راجع ملف: [`TEAM_TASK_DISTRIBUTION.md`](TEAM_TASK_DISTRIBUTION.md).

---

## 🌐 6. GitHub Repository Push Instructions

The repository is configured to point to your GitHub repo:
**[https://github.com/RemonAyman/advai](https://github.com/RemonAyman/advai)**

To commit and push all files to your remote repository, run the following commands in `E:\advai`:

```bash
git add .
git commit -m "feat: complete supervised classification project tasks 1-6 with extensive training, app, and docs"
git push -u origin main
```

*(If your remote branch already has an existing README or commit, pull and merge with `git pull origin main --rebase` before pushing).*

---

## 👨‍🔬 Authors & Acknowledgments
* **Dataset Credit**: Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (University of Wisconsin Hospitals).
* **Supervised Machine Learning Classification Project** | AdvAI Course.


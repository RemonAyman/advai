import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

def main():
    # Set styling
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8

    os.makedirs('figures', exist_ok=True)

    print("="*75)
    print("TASK 1: DATASET EXPLORATION")
    print("="*75)

    # Column names according to breast-cancer-wisconsin.names
    col_names = [
        'Sample_Code_Number',
        'Clump_Thickness',
        'Uniformity_Cell_Size',
        'Uniformity_Cell_Shape',
        'Marginal_Adhesion',
        'Single_Epithelial_Cell_Size',
        'Bare_Nuclei',
        'Bland_Chromatin',
        'Normal_Nucleoli',
        'Mitoses',
        'Class'
    ]

    data_path = os.path.join('breast+cancer+wisconsin+original', 'breast-cancer-wisconsin.data')
    df_raw = pd.read_csv(data_path, header=None, names=col_names)

    print(f"Dataset shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df_raw.head().to_string())

    print("\nData Types:")
    print(df_raw.dtypes)

    # Identify missing values represented as '?'
    missing_count_raw = (df_raw == '?').sum()
    print("\nMissing values represented by '?' per column:")
    print(missing_count_raw[missing_count_raw > 0])

    # Duplicates check
    exact_dup_count = df_raw.duplicated().sum()
    print(f"\nExact duplicate rows (including ID): {exact_dup_count}")

    # Class distribution
    print("\nClass distribution in raw data (2 = Benign, 4 = Malignant):")
    print(df_raw['Class'].value_counts())
    print("Class percentages:")
    print(df_raw['Class'].value_counts(normalize=True) * 100)

    print("\n" + "="*75)
    print("TASK 2: DATA PREPROCESSING")
    print("="*75)

    # 1. Replace '?' with NaN in Bare_Nuclei and convert to numeric
    df_clean = df_raw.copy()
    df_clean['Bare_Nuclei'] = pd.to_numeric(df_clean['Bare_Nuclei'].replace('?', np.nan))

    # 2. Handle duplicates: remove 8 exact duplicate records
    print(f"Rows before dropping duplicate records: {len(df_clean)}")
    df_clean = df_clean.drop_duplicates()
    print(f"Rows after dropping duplicate records: {len(df_clean)}")

    # 3. Remove irrelevant unique identifier
    df_clean = df_clean.drop(columns=['Sample_Code_Number'])
    print("Removed 'Sample_Code_Number' column (patient ID).")

    # 4. Map target variable: Benign (2) -> 0, Malignant (4) -> 1 (Malignant is positive class)
    df_clean['Class'] = df_clean['Class'].map({2: 0, 4: 1})
    print("Mapped target variable: Benign -> 0, Malignant -> 1 (Malignant is positive class).")

    # 5. Train-Test Split (80:20 ratio) with stratification to avoid data leakage
    X = df_clean.drop(columns=['Class'])
    y = df_clean['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Train class balance: Benign={sum(y_train==0)}, Malignant={sum(y_train==1)} ({y_train.mean():.1%} positive)")
    print(f"Test class balance: Benign={sum(y_test==0)}, Malignant={sum(y_test==1)} ({y_test.mean():.1%} positive)")

    # 6. Missing value imputation: SimpleImputer with median fit on X_train only
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)
    print(f"Median value learned from training set for 'Bare_Nuclei': {imputer.statistics_[5]}")
    print("Handled missing values using median imputation fit strictly on training set.")

    # 7. Standardization: StandardScaler fit on X_train only
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X.columns, index=X_test.index)
    print("Standardized features using StandardScaler fit strictly on training set.")

    print("\n" + "="*75)
    print("TASK 3: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS")
    print("="*75)

    # Chart 1: Missing Values Visualization
    plt.figure(figsize=(8, 4.5))
    missing_counts = df_raw.copy()
    missing_counts['Bare_Nuclei'] = pd.to_numeric(missing_counts['Bare_Nuclei'].replace('?', np.nan))
    null_series = missing_counts.isnull().sum()
    null_df = pd.DataFrame({'Feature': null_series.index, 'Missing_Count': null_series.values})
    ax = sns.barplot(data=null_df, x='Feature', y='Missing_Count', color='#c0392b')
    plt.title('Count of Missing Values per Feature (UCI Breast Cancer Original)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Features', fontsize=11, fontweight='bold')
    plt.ylabel('Missing Count', fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    for p in ax.patches:
        val = int(p.get_height())
        if val > 0:
            ax.annotate(f"{val} ({val/len(df_raw)*100:.1f}%)", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred', xytext=(0, 3),
                        textcoords='offset points')
    plt.tight_layout()
    plt.savefig('figures/1_missing_values.png', dpi=300)
    plt.close()
    print("Saved figures/1_missing_values.png")

    # Chart 2: Target Class Distribution Countplot
    plt.figure(figsize=(7, 5))
    class_labels = {0: 'Benign (Class 2)', 1: 'Malignant (Class 4)'}
    y_df = pd.DataFrame({'Class': y.map(class_labels)})
    ax = sns.countplot(data=y_df, x='Class', hue='Class', palette={'Benign (Class 2)': '#2980b9', 'Malignant (Class 4)': '#d35400'}, legend=False)
    plt.title('Target Class Distribution (Cleaned Dataset, N=691)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Tumor Diagnosis Class', fontsize=11, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=11, fontweight='bold')
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(f"{height} ({height/len(y)*100:.1f}%)", (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig('figures/2_target_distribution.png', dpi=300)
    plt.close()
    print("Saved figures/2_target_distribution.png")

    # Chart 3: Histograms of Feature Distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    feature_cols = list(X.columns)
    colors = sns.color_palette("tab10", len(feature_cols))

    for i, col in enumerate(feature_cols):
        sns.histplot(X_train_imp[col], kde=True, ax=axes[i], color=colors[i], bins=10, discrete=True)
        axes[i].set_title(f"{col.replace('_', ' ')}", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Value (Score 1-10)", fontsize=9)
        axes[i].set_ylabel("Frequency", fontsize=9)
        axes[i].set_xlim(0.5, 10.5)

    plt.suptitle("Feature Distributions (Training Set, N=552)", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('figures/3_feature_histograms.png', dpi=300)
    plt.close()
    print("Saved figures/3_feature_histograms.png")

    # Chart 4: Boxplots to compare feature distributions and identify outliers
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=X_train_imp, palette="Set2")
    plt.title("Boxplots of Cytological Features (Outlier & Distribution Analysis)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Cytological Features", fontsize=11, fontweight='bold')
    plt.ylabel("Value Range (1 to 10)", fontsize=11, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/4_feature_boxplots.png', dpi=300)
    plt.close()
    print("Saved figures/4_feature_boxplots.png")

    # Chart 5: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    train_combined = pd.concat([X_train_imp, y_train], axis=1)
    corr_matrix = train_combined.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title("Pearson Correlation Heatmap (Features and Diagnosis Class)", fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig('figures/5_correlation_heatmap.png', dpi=300)
    plt.close()
    print("Saved figures/5_correlation_heatmap.png")

    # Chart 6: Before-and-After Standardization Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sample_features = ['Clump_Thickness', 'Uniformity_Cell_Size', 'Marginal_Adhesion', 'Bland_Chromatin']

    for feat in sample_features:
        sns.kdeplot(X_train_imp[feat], ax=axes[0], label=feat.replace('_', ' '), fill=True, alpha=0.15)
    axes[0].set_title("Before Standardization (Original Scale: 1-10)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Original Feature Value", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].legend(loc='upper right', fontsize=9)

    for feat in sample_features:
        sns.kdeplot(X_train_scaled[feat], ax=axes[1], label=feat.replace('_', ' '), fill=True, alpha=0.15)
    axes[1].set_title("After Standardization (Zero Mean, Unit Variance)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Standardized Z-Score", fontsize=10)
    axes[1].set_ylabel("Density", fontsize=10)
    axes[1].legend(loc='upper right', fontsize=9)

    plt.suptitle("Comparison of Feature Distributions Before vs. After Standardization", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/6_before_after_standardization.png', dpi=300)
    plt.close()
    print("Saved figures/6_before_after_standardization.png")

    print("\n" + "="*75)
    print("TASK 4 & 5: MODEL IMPLEMENTATION, TUNING, AND EVALUATION")
    print("="*75)

    # 5-fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Models and hyperparameter grids
    model_configs = {
        'K-Nearest Neighbors (KNN)': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'Decision Tree (DT)': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 4, 5, 6, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Random Forest (RF)': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        },
        'Artificial Neural Network (ANN)': {
            'model': MLPClassifier(max_iter=1500, random_state=42),
            'params': {
                'hidden_layer_sizes': [(32,), (64,), (64, 32)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.001, 0.01, 0.1],
                'learning_rate_init': [0.001, 0.01]
            }
        },
        'Support Vector Machine (SVM)': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 5, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.01, 0.1]
            }
        }
    }

    trained_models = {}
    evaluation_results = []
    confusion_matrices = {}
    roc_data = {}

    for name, config in model_configs.items():
        print(f"\nTuning and Training {name}...")
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv,
            scoring='recall', # In medical diagnosis, high recall (sensitivity) minimizes false negatives
            n_jobs=2
        )
        grid.fit(X_train_scaled, y_train)
        best_estimator = grid.best_estimator_
        trained_models[name] = best_estimator
        
        # Test evaluation
        y_pred = best_estimator.predict(X_test_scaled)
        y_prob = best_estimator.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        confusion_matrices[name] = cm
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)
        
        evaluation_results.append({
            'Model': name,
            'Best Parameters': str(grid.best_params_),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        
        print(f"  Best Params  : {grid.best_params_}")
        print(f"  Accuracy     : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision    : {prec:.4f} ({prec*100:.2f}%)")
        print(f"  Recall       : {rec:.4f} ({rec*100:.2f}%)")
        print(f"  F1-Score     : {f1:.4f}")
        print(f"  ROC-AUC      : {auc:.4f}")

    # Comparison Table
    comparison_df = pd.DataFrame(evaluation_results)
    print("\n" + "="*75)
    print("ALL MODELS COMPARISON TABLE (TASK 5)")
    print("="*75)
    summary_table = comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    print(summary_table.to_string(index=False))
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print("\nSaved model_comparison_results.csv")

    # Chart 7: Confusion Matrices for All 5 Models
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    model_short_names = [
        'K-Nearest Neighbors (KNN)',
        'Decision Tree (DT)',
        'Random Forest (RF)',
        'Artificial Neural Network (ANN)',
        'Support Vector Machine (SVM)'
    ]

    for idx, m_name in enumerate(model_short_names):
        cm = confusion_matrices[m_name]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    annot_kws={'size': 14, 'weight': 'bold'},
                    xticklabels=['Benign (0)', 'Malignant (1)'],
                    yticklabels=['Benign (0)', 'Malignant (1)'])
        axes[idx].set_title(f"{m_name}\nAcc: {comparison_df.loc[comparison_df['Model']==m_name, 'Accuracy'].values[0]:.3f} | Recall: {comparison_df.loc[comparison_df['Model']==m_name, 'Recall'].values[0]:.3f}",
                            fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=9)
        axes[idx].set_ylabel('True Label', fontsize=9)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle("Confusion Matrices for All Five Models (Test Set N=139, Positive=Malignant)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('figures/7_all_confusion_matrices.png', dpi=300)
    plt.close()
    print("Saved figures/7_all_confusion_matrices.png")

    # Chart 8: ROC Curves Comparison
    plt.figure(figsize=(9, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        plt.plot(fpr, tpr, lw=2.2, color=colors[i], label=f"{name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curves Comparison', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig('figures/8_roc_curves_comparison.png', dpi=300)
    plt.close()
    print("Saved figures/8_roc_curves_comparison.png")

    # Chart 9: Metrics Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    metrics_plot_df = comparison_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                                         var_name='Metric', value_name='Score')
    ax = sns.barplot(data=metrics_plot_df, x='Model', y='Score', hue='Metric', palette='viridis')
    plt.title('Cross-Model Performance Metrics Comparison on Unseen Test Set', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Classification Algorithm', fontsize=11, fontweight='bold')
    plt.ylabel('Metric Score (0.0 - 1.0)', fontsize=11, fontweight='bold')
    plt.ylim(0.80, 1.02)
    plt.xticks(rotation=20, ha='right', fontsize=10)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('figures/9_metrics_comparison_bar.png', dpi=300)
    plt.close()
    print("Saved figures/9_metrics_comparison_bar.png")

    # Chart 10: Random Forest Feature Importance
    rf_best = trained_models['Random Forest (RF)']
    importances = rf_best.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)

    plt.figure(figsize=(9, 5))
    feat_importances.plot(kind='barh', color='#2ca02c')
    plt.title('Random Forest Cytological Feature Importance (Gini)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Importance Score', fontsize=11, fontweight='bold')
    plt.ylabel('Cytological Feature', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/10_rf_feature_importance.png', dpi=300)
    plt.close()
    print("Saved figures/10_rf_feature_importance.png")

    print("\nALL TASKS EXECUTED SUCCESSFULLY!")

if __name__ == '__main__':
    main()

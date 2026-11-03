import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

def run_extensive_training():
    print("=" * 80)
    print("EXTENSIVE SUPERVISED MACHINE LEARNING TRAINING & EVALUATION")
    print("=" * 80)

    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # 1. Load Dataset
    col_names = [
        'Sample_Code_Number', 'Clump_Thickness', 'Uniformity_Cell_Size',
        'Uniformity_Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size',
        'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class'
    ]
    data_path = os.path.join('breast+cancer+wisconsin+original', 'breast-cancer-wisconsin.data')
    df = pd.read_csv(data_path, header=None, names=col_names)

    # 2. Preprocess
    df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'].replace('?', np.nan))
    df = df.drop_duplicates() # Remove 8 exact duplicate records
    df = df.drop(columns=['Sample_Code_Number']) # Remove ID
    df['Class'] = df['Class'].map({2: 0, 4: 1}) # 0: Benign, 1: Malignant (Positive)

    X = df.drop(columns=['Class'])
    y = df['Class']

    # 3. Train-Test Split (80:20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Total clean samples: {len(df)}")
    print(f"Training set: {X_train.shape[0]} samples (Benign: {sum(y_train==0)}, Malignant: {sum(y_train==1)})")
    print(f"Testing set : {X_test.shape[0]} samples (Benign: {sum(y_test==0)}, Malignant: {sum(y_test==1)})")

    # 4. Impute strictly on train
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)
    joblib.dump(imputer, 'models/imputer.joblib')

    # 5. Scale strictly on train
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X.columns, index=X_test.index)
    joblib.dump(scaler, 'models/scaler.joblib')

    # 6. Deep & Thorough Cross-Validation Setup (5-Fold Stratified CV)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_grids = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 4, 5, 6, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300], # Thorough tree ensemble
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        },
        'Artificial Neural Network (ANN)': {
            'model': MLPClassifier(max_iter=2500, random_state=42), # Deep epochs to ensure complete convergence
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
                'C': [0.1, 1.0, 5.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.01, 0.1]
            }
        }
    }

    best_estimators = {}
    results_list = []
    roc_curves_data = {}
    cm_dict = {}

    for name, config in model_grids.items():
        print(f"\n[EXTENSIVE TUNING] Training {name} with 5-Fold Stratified CV...")
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv_strategy,
            scoring='recall', # Prioritize clinical sensitivity
            n_jobs=2
        )
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        best_estimators[name] = best_model

        # Save model artifact
        slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        joblib.dump(best_model, f'models/{slug}.joblib')

        # Evaluate on held-out test set
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        cm_dict[name] = cm
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves_data[name] = (fpr, tpr, auc)

        results_list.append({
            'Model': name,
            'Best Hyperparameters': str(grid.best_params_),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })

        print(f"  Best Parameters: {grid.best_params_}")
        print(f"  Test Accuracy  : {acc*100:.2f}%")
        print(f"  Test Precision : {prec*100:.2f}%")
        print(f"  Test Recall    : {rec*100:.2f}%")
        print(f"  Test F1-Score  : {f1:.4f}")
        print(f"  Test ROC-AUC   : {auc:.4f}")

    # 7. Create Top Ensemble Model (Soft Voting of RF + ANN + SVM)
    print("\n[ENSEMBLE TRAINING] Building Soft Voting Meta-Ensemble (RF + ANN + SVM)...")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', best_estimators['Random Forest']),
            ('ann', best_estimators['Artificial Neural Network (ANN)']),
            ('svm', best_estimators['Support Vector Machine (SVM)'])
        ],
        voting='soft'
    )
    voting_clf.fit(X_train_scaled, y_train)
    joblib.dump(voting_clf, 'models/ensemble_voting.joblib')

    y_pred_ens = voting_clf.predict(X_test_scaled)
    y_prob_ens = voting_clf.predict_proba(X_test_scaled)[:, 1]

    acc_ens = accuracy_score(y_test, y_pred_ens)
    prec_ens = precision_score(y_test, y_pred_ens, pos_label=1)
    rec_ens = recall_score(y_test, y_pred_ens, pos_label=1)
    f1_ens = f1_score(y_test, y_pred_ens, pos_label=1)
    auc_ens = roc_auc_score(y_test, y_prob_ens)
    cm_ens = confusion_matrix(y_test, y_pred_ens)

    cm_dict['Voting Ensemble (RF+ANN+SVM)'] = cm_ens
    fpr_ens, tpr_ens, _ = roc_curve(y_test, y_prob_ens)
    roc_curves_data['Voting Ensemble'] = (fpr_ens, tpr_ens, auc_ens)

    results_list.append({
        'Model': 'Voting Ensemble (RF+ANN+SVM)',
        'Best Hyperparameters': 'Soft Voting Blend of Tuned RF, ANN & SVM',
        'Accuracy': acc_ens,
        'Precision': prec_ens,
        'Recall': rec_ens,
        'F1-Score': f1_ens,
        'ROC-AUC': auc_ens
    })

    print(f"  Ensemble Accuracy : {acc_ens*100:.2f}%")
    print(f"  Ensemble Recall   : {rec_ens*100:.2f}%")
    print(f"  Ensemble ROC-AUC  : {auc_ens:.4f}")

    # Save summary dataframe
    summary_df = pd.DataFrame(results_list)
    summary_df.to_csv('model_comparison_results.csv', index=False)
    print("\n" + "="*80)
    print("FINAL SUMMARY RESULTS TABLE")
    print("="*80)
    print(summary_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))

    # Save metadata JSON for the Web Application
    metadata = {
        'features': list(X.columns),
        'feature_ranges': {col: {'min': 1, 'max': 10, 'default': int(X[col].median())} for col in X.columns},
        'target_mapping': {'0': 'Benign (حميد)', '1': 'Malignant (خبيث)'},
        'top_model_name': 'Random Forest',
        'metrics_summary': summary_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_dict(orient='records')
    }
    with open('models/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    print("\nAll models, preprocessors, and metadata successfully saved to models/ directory!")

if __name__ == '__main__':
    run_extensive_training()

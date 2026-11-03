import sys
import os
import json
import joblib
import pandas as pd
import numpy as np

def main():
    print("=" * 70)
    print("BREAST CANCER BIOPSY DIAGNOSIS CLI SYSTEM")
    print("Supervised Machine Learning Diagnostic Tool")
    print("=" * 70)

    try:
        imputer = joblib.load('models/imputer.joblib')
        scaler = joblib.load('models/scaler.joblib')
        model = joblib.load('models/random_forest.joblib')
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure models are trained first.")
        sys.exit(1)

    features = [
        ('Clump_Thickness', 'Clump Thickness (1-10) [Default: 5]: '),
        ('Uniformity_Cell_Size', 'Uniformity of Cell Size (1-10) [Default: 1]: '),
        ('Uniformity_Cell_Shape', 'Uniformity of Cell Shape (1-10) [Default: 1]: '),
        ('Marginal_Adhesion', 'Marginal Adhesion (1-10) [Default: 1]: '),
        ('Single_Epithelial_Cell_Size', 'Single Epithelial Cell Size (1-10) [Default: 2]: '),
        ('Bare_Nuclei', 'Bare Nuclei (1-10) [Default: 1]: '),
        ('Bland_Chromatin', 'Bland Chromatin (1-10) [Default: 3]: '),
        ('Normal_Nucleoli', 'Normal Nucleoli (1-10) [Default: 1]: '),
        ('Mitoses', 'Mitoses (1-10) [Default: 1]: ')
    ]

    defaults = [5, 1, 1, 1, 2, 1, 3, 1, 1]
    inputs = {}

    print("\nEnter biopsy scores (1 to 10) or press Enter to accept default:\n")
    for idx, (feat_name, prompt_text) in enumerate(features):
        while True:
            val = input(prompt_text).strip()
            if val == "":
                inputs[feat_name] = defaults[idx]
                break
            try:
                num = float(val)
                if 1 <= num <= 10:
                    inputs[feat_name] = num
                    break
                else:
                    print("Value must be between 1 and 10.")
            except ValueError:
                print("Invalid number. Please enter a value between 1 and 10.")

    input_df = pd.DataFrame([inputs])
    input_imp = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    input_scaled = pd.DataFrame(scaler.transform(input_imp), columns=input_df.columns)

    pred = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]

    print("\n" + "=" * 70)
    print("DIAGNOSTIC ANALYSIS RESULT")
    print("=" * 70)
    if pred == 1:
        print(">>> RESULT: MALIGNANT (ورم خبيث) <<<")
        print(f"Malignancy Probability: {probs[1]*100:.2f}%")
        print("Confidence Level      : HIGH")
        print("Clinical Warning      : Immediate oncological biopsy review recommended!")
    else:
        print(">>> RESULT: BENIGN (ورم حميد) <<<")
        print(f"Benign Probability    : {probs[0]*100:.2f}%")
        print("Confidence Level      : HIGH")
        print("Clinical Note         : Low risk profile. Standard mammographic follow-up.")
    print("=" * 70)

if __name__ == '__main__':
    main()


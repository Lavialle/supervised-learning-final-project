# 🦠 Multi-Resistance Antibiotic Susceptibility — Supervised Learning Final Project

![Bacteria_image](./assets/bacteria_banner.jpg)

## Contributors

- Côme Bonneviale
- Ange Lavialle

---

## Business Challenge

Antibiotic resistance is a major public health issue. Predicting whether a patient with a bacterial strain will be multi-drug resistant (MDR) knowing is susceptibility to the most given antibiotics (ctx/cro) helps clinicians anticipate treatments, limit spread, and improve patient care.  
This project aims to build a machine learning model to predict MDR status from clinical and biological data.

**Target Calculation:**
The target variable is_MDR was engineered according to clinical guidelines. A strain is considered MDR if it is resistant to at least one antibiotic in three or more distinct antibiotic families.

The antibiotic families and their corresponding columns in the cleaned dataset are:

- beta_lactams
  - `amx/amp` → Amoxicillin / Ampicillin (aminopenicillins)
  - `amc` → Amoxicillin + Clavulanic Acid (β-lactam/β-lactamase inhibitor)
  - `cz` → Cefazolin (first-generation cephalosporin)
  - `fox` → Cefoxitin (second-generation cephamycin)
  - `ctx/cro` → Cefotaxime / Ceftriaxone (third-generation cephalosporins)
  - `ipm` → Imipenem (a carbapenem, considered a last-resort treatment)
- aminosides
  - `gen` → Gentamicin
  - `an` → Amikacin (broader spectrum, often effective against gentamicin-resistant strains)
- quinolones
  - `acide_nalidixique` → Nalidixic Acid (first-generation quinolone, Gram-negative coverage)
  - `ofx` → Ofloxacin (fluoroquinolone, broad spectrum)
  - `cip` → Ciprofloxacin (fluoroquinolone, widely used for urinary tract infections)
- phenicols
  - `c` → Chloramphenicol (broad-spectrum, rarely used due to toxicity but still tested)
- nitrofuranes
  - `furanes` → Nitrofurantoin (commonly used for urinary tract infections)
- sulfamides
  - `co-trimoxazole` → Combination of Trimethoprim + Sulfamethoxazole (folate pathway inhibitor)
- polymyxins
  - `colistine` → Colistin (polymyxin class, last-line therapy for carbapenem-resistant organisms)

---

## Dataset

- **Source:** `data/Bacteria_dataset_Multiresictance.csv` from [Kaggle](https://www.kaggle.com/datasets/adilimadeddinehosni/multi-resistance-antibiotic-susceptibility)
- **Rows:** 10,710
- **Columns:** 27
- **Features:** Patient info (age, gender, comorbidities), strain info (name, code, collection date), resistance to ctx/cro.
- **Target:** `is_MDR` (computed from resistance columns using business rules).

---

## Project Architecture

```
supervised-learning-final-project/
│
├── data/
│   └── Bacteria_dataset_Multiresictance.csv    # Raw dataset
│   └── cleaned_bacteria_dataset.csv            # Clean version for EDA
├── main.py                                     # Main module for code execution
├── utils/                                      # Tools and modules to preprocessing and model evaluation
│   └── preprocessing.py
│   └── evaluation.py
│   └── utils.py                                # Cleaning functions
├── models/                                     # Different experiments
│   └── hyperopt_model.py
│   └── individual_model.py
│   └── stacking_model.py
├── requirements.txt
├── README.md
├── eda.ipynb                                   # Notebook for the Exploratory Data Analysis
├── figures/                                  # Results for each models
│   └── ...
└── assets/
    └── bacteria_banner.jpg
```

---

## Pipeline Overview (example for the Stacking Model)

```mermaid
flowchart TD
    A[Raw Dataset] --> B[Global Cleaning: Missing Values, Date Parsing]
    B --> C[Feature Engineering: Frequency Encoding, Interaction Terms]
    C --> D[Train/Test Split]
    D --> E[Preprocessing: ColumnTransformer]
    E --> F1[XGBoost]
    E --> F2[LightGBM]
    E --> F3[CatBoost]
    F1 --> G[Stacked Model: Meta-Model: Random Forest]
    F2 --> G
    F3 --> G
    G --> H[Prediction: MDR or Non-MDR]
    H --> I[Evaluation: F1-Score, Decision Threshold Optimization]
    H --> J[MLflow Logging]
```

---

## Reproducibility Instructions

1. **Copy the dataset:**  
   Place `Bacteria_dataset_Multiresictance.csv` in the `data/` folder.

2. **Python version:**  
   Use **Python 3.10** (recommended).

3. **Virtual Environment Creation:**

   ```sh
   python -m venv venv
   source venv/Scripts/activate
   ```

4. **Run the pipeline and chose a model you want to test:**
   ```sh
   python main.py --model [individual, hyperopt, stacking]
   ```
   > Check if you have `uv` installed in your global environment before running the main.py pipeline.

---

## Baseline

- **Features:** age, gender, infection frequency, comorbidities (diabetes, hypertension, prior hospitalization), strain.
- **Pre-processing:** Column cleaning, missing value handling, OneHot encoding for categoricals, scaling for numericals, imputation.
- **Model:** RandomForestClassifier, CatBoostClassifier, LogisticRegression, no hyperparameter optimization.
- **Metric:** F1-score (stratified 5-fold CV).
- **Score:** _[F1 = 0.59 ± 0.03]_

---

## Experiment Tracking

- **Hyperparameter optimization:** Used Hyperopt for CatBoost, XGBoost, and stacking.
- **Stacking:** Combined multiple models with RandomForest as meta-model.
- **Feature engineering:** Added resistance-based features.
- **Impact:** _[Fill in score improvements, e.g. stacking F1 = 0.76 ± 0.02]_

---

## Next Steps

- Add final scores, feature choices, and detailed experiment impacts.
- Add full contributor names.
- Insert pipeline diagrams and suggestive images for visual enhancement.

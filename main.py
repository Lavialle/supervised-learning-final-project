"""This module is going to have several objectives.
1. Load the dataset from a CSV file.
2. Preprocess the data using Pipeline module from sklearn.
3. Train a machine learning model.
4. Evaluate the model's performance.
5. Save the trained model to a file.
"""

# 0. Get the requirements from requirements.txt and install them using pip if not already installed.

# with open("requirements.txt") as f:
#     requirements = f.readlines()
# import subprocess
# for requirement in requirements:
#     subprocess.check_call(["pip", "install", requirement.strip()])


# Import necessary librairies
import utils  # custom utility functions for data preprocessing

import pandas as pd # for data manipulation
import numpy as np # for numerical operations
from ydata_profiling import ProfileReport # for generating data profile reports
from pathlib import Path # for handling file paths
from IPython.display import display # for displaying dataframes
from sklearn.model_selection import train_test_split, cross_val_score



from sklearn.exceptions import NotFittedError # for handling exceptions
from sklearn.pipeline import Pipeline # for creating machine learning pipelines
from sklearn.compose import ColumnTransformer # for column-wise transformations
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder # for data preprocessing
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer # for imputing missing values
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

class StopExecution(Exception):
    def _render_traceback_(self):
        return []

# 1. load the dataset

RAW_BACTERIA_RESISTANCE_PATH = Path("./data/Bacteria_dataset_Multiresictance.csv")

if not RAW_BACTERIA_RESISTANCE_PATH.exists():
    print("Could not find the dataset at path:", RAW_BACTERIA_RESISTANCE_PATH)
    raise StopExecution

try:
    RAW_BACTERIA_RESISTANCE_DF = pd.read_csv(RAW_BACTERIA_RESISTANCE_PATH)
except Exception:
    raise StopExecution(
        f"""Something went wrong while loading the dataset.
You probably made a mistake while copying the dataset file on your machine:
{RAW_BACTERIA_RESISTANCE_PATH.resolve()}"""
    )

if RAW_BACTERIA_RESISTANCE_DF.shape != (10710, 27):
    raise StopExecution(
        f"""The dataset shape is incorrect: {RAW_BACTERIA_RESISTANCE_DF.shape}.
You probably made a mistake while copying the dataset file on your machine:
{RAW_BACTERIA_RESISTANCE_PATH.resolve()}"""
    )

print("Dataset loaded successfully, here is a preview of the first 10 lines:")
display(RAW_BACTERIA_RESISTANCE_DF.head(n=10))

# 2. Preprocess the data



# Construction  of the preprocessing pipeline
# ========== STEP 1: Global Cleaning Pipeline ==========

def global_cleaning(df: pd.DataFrame, drop_columns=None) -> pd.DataFrame:
    """
    Nettoyage complet du dataset :
    - Nettoyage des noms de colonnes
    - Normalisation des valeurs manquantes
    - Split colonnes age/gender et souches
    - Nettoyage et normalisation des noms de souches
    - Uniformisation des valeurs de susceptibilité aux antibiotiques
    - Suppression des doublons
    - Normalisation des colonnes booléennes
    - Nettoyage et imputation des dates
    - Suppression des lignes entièrement NaN (sauf colonnes optionnelles)
    - Cast des colonnes catégorielles et booléennes
    - Imputation des colonnes numériques
    - Suppression optionnelle de colonnes
    """
    df = df.copy()

    # Cleaning and general preprocessing steps
    df = utils.clean_column_names(df)
    df = utils.normalize_missing_values(df)
    df = utils.split_age_gender(df)
    df = utils.split_id_strain(df)
    df = utils.clean_strain_names(df)
    df = utils.normalize_strain_names(df)
    df = utils.uniformize_susceptibility_values(df)
    df = utils.drop_duplicates(df)
    df = utils.normalize_boolean_columns(df)
    df = utils.clean_collection_date(df)
    df = utils.handle_missing_dates(df, date_col="collection_date")  # imputation dates

    # Empty rows removal
    df = utils.drop_all_nan_rows(df)

    # Cast columns
    df = utils.cast_boolean_columns(df)
    df = utils.cast_categorical_columns(df)
    df = utils.cast_numerical_columns(df)

    # Optional drop columns
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors='ignore')

    return df


# ==============================================================================================================================

# PROFILE_REPORT_PATH = Path("profile.html")
# if PROFILE_REPORT_PATH.exists():
#     PROFILE_REPORT_PATH.unlink(missing_ok=True)
#     print("Existing profile report removed")
# generate_profile_report(RAW_BACTERIA_RESISTANCE_DF, PROFILE_REPORT_PATH)
# if not PROFILE_REPORT_PATH.exists():
#     raise StopExecution(
#         f"The profile report was not generated at {PROFILE_REPORT_PATH}"
#     )

cleaning_step = FunctionTransformer(global_cleaning, kw_args={'drop_columns': ['id', 'name', 'address', 'notes', 'email', 'collection_date']}, validate=False)
cleaned_df = cleaning_step.transform(RAW_BACTERIA_RESISTANCE_DF)

# Exporting to CSV for EDA
EDA_OUTPUT_PATH = Path("./data/cleaned_bacteria_dataset.csv")
if not EDA_OUTPUT_PATH.exists():
    cleaned_df.to_csv(EDA_OUTPUT_PATH, index=False)
    print(f"Cleaned dataset exported for EDA: {EDA_OUTPUT_PATH.resolve()}")
else:
    print("Existing cleaned dataset for EDA found")

# # Generate a profile report to check the cleaned data
# PROFILE_REPORT_PATH = Path("cleaned_profile.html")
# if PROFILE_REPORT_PATH.exists():
#     PROFILE_REPORT_PATH.unlink(missing_ok=True)
#     print("Existing profile report removed")
# generate_profile_report(cleaned_df, PROFILE_REPORT_PATH, title="Profiling Report on Bacteria Dataset Cleaned")
# if not PROFILE_REPORT_PATH.exists():
#     raise StopExecution(
#         f"The profile report was not generated at {PROFILE_REPORT_PATH}"
#     )


print("Dataset cleaned successfully, here is a preview of the first 10 lines:")
print(cleaned_df.head(10))
print(cleaned_df.dtypes)
print(cleaned_df.isna().sum())
print(cleaned_df.shape)


# ========== STEP 2: ColumnTransformer for scaling and encoding ==========

# numerical_cols, boolean_cols, categorical_cols = get_column_types(cleaned_df)
numerical_cols = ['age', 'infection_freq']
categorical_cols = ['gender', 'strain_norm']
boolean_cols = ['diabetes', 'hypertension', 'hospital_before']


# Encoders / scalers
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

boolean_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessing = ColumnTransformer([
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols),
    ('bool', boolean_transformer, boolean_cols)
])

target_cols = [
    'amx/amp_norm', 'amc_norm', 'cz_norm', 'fox_norm', 
    'ctx/cro_norm', 'ipm_norm', 'gen_norm', 'an_norm',
    'acide_nalidixique_norm', 'ofx_norm', 'cip_norm', 
    'c_norm', 'co-trimoxazole_norm', 'furanes_norm', 'colistine_norm'
]

X = cleaned_df.drop(columns=target_cols)
y = cleaned_df[target_cols]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.fillna(y_train.mode().iloc[0])
y_test = y_test.fillna(y_train.mode().iloc[0])

ml_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('model', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
])

# Fit sur le train


ml_pipeline.fit(X_train, y_train)

# Score sur le test
test_score = ml_pipeline.score(X_test, y_test)
print(f"Test score: {test_score:.3f}")

# Cross-validation sur le train
# from sklearn.metrics import classification_report

# y_pred = ml_pipeline.predict(X_test)
# print(classification_report(y_test, y_pred))

# cv_scores = cross_val_score(ml_pipeline, X_train, y_train, cv=5, scoring='accuracy')
# print(f"CV scores: {cv_scores}")
# print(f"CV mean: {cv_scores.mean():.3f}")

# # Train a model for each target


models = {}

for target in target_cols:
    print(f"Training model for {target}...")
    model = Pipeline([
        ('preprocess', preprocessing),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    models[target] = model
    model.fit(X_train, y_train[target])
    score = model.score(X_test, y_test[target])
    print(f"Test score for {target}: {score:.3f}")
# # Test multi-model


# multi_model = Pipeline([
#     ('cleaning', cleaning_step),
#     ('preprocess', preprocessing),
#     ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
# ])

# # One vs Rest Classifier example



# ovr_model = Pipeline([
#     ('preprocess', preprocessing),
#     ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
# ])
# ovr_model.fit(X_train, y_train)
# ovr_score = ovr_model.score(X_test, y_test)
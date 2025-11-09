"""This module is going to have several objectives.
1. Load the dataset from a CSV file.
2. Preprocess the data using Pipeline module from sklearn.
3. Train a machine learning model.
4. Evaluate the model's performance.
5. Save the trained model to a file.
"""

# 0. Get the requirements from requirements.txt and install them using pip if not already installed.


# Import necessary librairies
import utils  # custom utility functions for data preprocessing
import mlflow
import mlflow.sklearn
import pandas as pd # for data manipulation
import numpy as np # for numerical operations
from pathlib import Path # for handling file paths
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from sklearn.exceptions import NotFittedError # for handling exceptions
from sklearn.pipeline import Pipeline # for creating machine learning pipelines
from sklearn.compose import ColumnTransformer # for column-wise transformations
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder # for data preprocessing
from sklearn.impute import SimpleImputer # for imputing missing values

# Model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

classification_models = {
    'XGBoost': XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, use_label_encoder=False),
    'LightGBM': LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31),
    'CatBoost': CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0)
}

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

print("Dataset loaded successfully!")


# 2. Preprocess the data



# Construction  of the preprocessing pipeline
# ========== STEP 1: Global Cleaning Pipeline ==========

def global_cleaning(df: pd.DataFrame, drop_columns=None) -> pd.DataFrame:
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
    # Empty rows removal
    df = utils.drop_all_nan_rows(df)
    # Optional drop columnss
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors='ignore')

    return df



# PROFILE_REPORT_PATH = Path("profile.html")
# if PROFILE_REPORT_PATH.exists():
#     PROFILE_REPORT_PATH.unlink(missing_ok=True)
#     print("Existing profile report removed")
# utils.generate_profile_report(RAW_BACTERIA_RESISTANCE_DF, PROFILE_REPORT_PATH)
# if not PROFILE_REPORT_PATH.exists():
#     raise StopExecution(
#         f"The profile report was not generated at {PROFILE_REPORT_PATH}"
#     )


# Generate a CSV file for EDA
eda_step = FunctionTransformer(global_cleaning, validate=False)
eda_df = eda_step.transform(RAW_BACTERIA_RESISTANCE_DF)

EDA_OUTPUT_PATH = Path("./data/cleaned_bacteria_dataset.csv")
if not EDA_OUTPUT_PATH.exists():
    eda_df.to_csv(EDA_OUTPUT_PATH, index=False)
    print(f"Cleaned dataset exported for EDA: {EDA_OUTPUT_PATH.resolve()}")
else:
    print("Existing cleaned dataset for EDA found")


# ========== STEP 2: ColumnTransformer for scaling and encoding ==========

# numerical_cols, boolean_cols, categorical_cols = get_column_types(cleaned_df)
numerical_cols = ['age', 'infection_freq']
boolean_cols = ['diabetes', 'hypertension', 'hospital_before']
categorical_cols = ['gender',  'strain']


# Encoders / scalers
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
])

boolean_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

def add_resistance_features(df: pd.DataFrame, drop_columns=None) -> pd.DataFrame:
    df = df.copy()
    df = utils.compute_family_resistance(df)
    df = utils.compute_antibiotic_resistance(df)
    df = utils.compute_is_MDR(df)
    df = utils.cast_boolean_columns(df)
    df = utils.cast_categorical_columns(df)
    df = utils.cast_numerical_columns(df)
    df = utils.drop_correlated_features(df)
    df = utils.drop_nan_rows(df)
    df = utils.drop_duplicates(df)
    return df


cleaning = FunctionTransformer(global_cleaning, kw_args={'drop_columns': ['id', 'name', 'address', 'notes', 'email', 'collection_date']}, validate=False)
feature_engineering = FunctionTransformer(add_resistance_features, validate=False)
preprocessing = ColumnTransformer([
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols),
    ('bool', boolean_transformer, boolean_cols)
])


cleaning_engineering_pipeline = Pipeline([
    ('cleaning', cleaning),
    ('feature_engineering', feature_engineering)
])

RandomForest = Pipeline([
    ('preprocess', preprocessing),
    ('RandomForest', RandomForestClassifier(random_state=42))
])

CatBoost = Pipeline([
    ('preprocess', preprocessing),
    ('CatBoost', CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0))
])

XGBoost = Pipeline([
    ('preprocess', preprocessing),
    ('XGBoost', XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic'))
])

Logistic = Pipeline([
    ('preprocess', preprocessing),
    ('Logistic', LogisticRegression())        
])

HistGradient = Pipeline([
    ('preprocess', preprocessing),
    ('HistGradient', HistGradientBoostingClassifier(random_state=42))    
])

df = cleaning_engineering_pipeline.transform(RAW_BACTERIA_RESISTANCE_DF)

# PROFILE_REPORT_PATH = Path("cleaned_engineered_profile.html")
# if PROFILE_REPORT_PATH.exists():
#     PROFILE_REPORT_PATH.unlink(missing_ok=True)
#     print("Existing profile report removed")
# utils.generate_profile_report(df, PROFILE_REPORT_PATH, title="Profiling Report on Bacteria Dataset Cleaned and Engineered")
# if not PROFILE_REPORT_PATH.exists():
#     raise StopExecution(
#         f"The profile report was not generated at {PROFILE_REPORT_PATH}"
#     )
# print(df.shape)

# Definition of the features and target dataset. The goal is to predict if a new strain will be MDR.
X = df.drop(columns=["is_MDR"])
y = df["is_MDR"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


for model, name in zip([RandomForest, CatBoost, XGBoost], ["RF", "CB", "XG"]):
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    print(f"{name} F1: {np.mean(scores):.3f} ± {np.std(scores):.3f}")


RandomForest.fit(X_train, y_train)
rf_pred = RandomForest.predict(X_test)
cm_rf = confusion_matrix(y_test, rf_pred)
print(cm_rf)
print(classification_report(y_test, rf_pred))

CatBoost.fit(X_train, y_train)
cb_pred = CatBoost.predict(X_test)
cm_cb = confusion_matrix(y_test, cb_pred)
print(cm_cb)
print(classification_report(y_test, cb_pred))

XGBoost.fit(X_train, y_train)
xg_pred = XGBoost.predict(X_test)
cm_xg = confusion_matrix(y_test, xg_pred)
print(cm_xg)
print(classification_report(y_test, xg_pred))


param_grid = {
    "HistGradient__learning_rate": [0.01, 0.05, 0.1],
    "HistGradient__max_iter": [100, 200],
    "HistGradient__max_depth": [3, 5, None]
}

grid = GridSearchCV(HistGradient, param_grid, scoring='f1', cv=5, n_jobs=-1)

# MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("bacteria_resistance_classification")

with mlflow.start_run() as run:
    mlflow.set_tag("model_type", "HistGradientBoostingClassifier")

    # Train
    grid.fit(X_train, y_train)

    # Best model
    best_model = grid.best_estimator_
    mlflow.log_params(grid.best_params_)

    # Predict
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("f1_score", f1)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    report = classification_report(y_test, y_pred)
    print(report)

    # Optionally log the report as text
    mlflow.log_text(report, "classification_report.txt")

    # Log the model
    mlflow.sklearn.log_model(best_model, artifact_path="model")
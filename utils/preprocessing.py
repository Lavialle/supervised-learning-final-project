"""
Data preprocessing utilities for the Bacteria Resistance dataset.
"""

import utils.utils as utils
import pandas as pd # for data manipulation
import numpy as np # for numerical operations
from pathlib import Path # for handling file paths
from sklearn.pipeline import Pipeline # for creating machine learning pipelines
from sklearn.compose import ColumnTransformer # for column-wise transformations
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder # for data preprocessing
from sklearn.impute import SimpleImputer # for imputing missing values
from sklearn.base import BaseEstimator, TransformerMixin # for custom transformers

class StopExecution(Exception):
    def _render_traceback_(self):
        return []

def load_data():
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
    return RAW_BACTERIA_RESISTANCE_DF

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
# eda_step = FunctionTransformer(global_cleaning, validate=False)
# eda_df = eda_step.transform(RAW_BACTERIA_RESISTANCE_DF)

# EDA_OUTPUT_PATH = Path("./data/cleaned_bacteria_dataset.csv")
# if not EDA_OUTPUT_PATH.exists():
#     eda_df.to_csv(EDA_OUTPUT_PATH, index=False)
#     print(f"Cleaned dataset exported for EDA: {EDA_OUTPUT_PATH.resolve()}")
# else:
#     print("Existing cleaned dataset for EDA found")


# ========== STEP 2: ColumnTransformer for scaling and encoding ==========

# numerical_cols, boolean_cols, categorical_cols = get_column_types(cleaned_df)
numerical_cols = ['infection_freq', 'age_comorb']
boolean_cols = ['ctx/cro_resistant']#['diabetes', 'hypertension', 'hospital_before']
categorical_cols = ['gender', 'age_bin']
frequency_cols = ['strain']

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_map = None

    def fit(self, X, y=None):
        # X peut être DataFrame, Series ou array
        if isinstance(X, pd.DataFrame):
            s = X['strain']
        elif isinstance(X, pd.Series):
            s = X
        elif isinstance(X, np.ndarray):
            # Si array 2D, on le squeeze
            if X.ndim == 2 and X.shape[1] == 1:
                X = X.ravel()
            s = pd.Series(X)
        else:
            raise ValueError("Unsupported input type for FrequencyEncoder")
        self.freq_map = s.value_counts(normalize=True)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            s = X['strain']
        elif isinstance(X, pd.Series):
            s = X
        elif isinstance(X, np.ndarray):
            if X.ndim == 2 and X.shape[1] == 1:
                X = X.ravel()
            s = pd.Series(X)
        else:
            raise ValueError("Unsupported input type for FrequencyEncoder")
        return s.map(self.freq_map).to_frame(name='strain_freq')
    
    def get_feature_names_out(self, input_features=None):
        # Retourne le nom de la feature créée
        return np.array(['strain_freq'])


# Encoders / scalers
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

frequency_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('freq_encoder', FrequencyEncoder())
])

boolean_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

def add_resistance_features(df: pd.DataFrame, drop_columns=None) -> pd.DataFrame:
    df = df.copy()
    df = utils.compute_family_resistance(df)
    df = utils.compute_antibiotic_resistance(df)
    df = utils.compute_is_MDR(df)
    df = utils.add_age_comorbidity_interaction(df)
    df = utils.bin_age_and_drop(df)
    df = utils.cast_boolean_columns(df)
    df = utils.cast_categorical_columns(df)
    df = utils.cast_numerical_columns(df)
    df = utils.drop_correlated_features(df)
    df = utils.drop_nan_rows(df)
    df = utils.drop_duplicates(df)
    return df


cleaning = FunctionTransformer(global_cleaning, kw_args={'drop_columns': ['id', 'name', 'address', 'notes', 'email', 'collection_date']}, validate=False)
feature_engineering = FunctionTransformer(add_resistance_features, validate=False)

cleaning_engineering_pipeline = Pipeline([
    ('cleaning', cleaning),
    ('feature_engineering', feature_engineering)
])

preprocessing = ColumnTransformer([
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols),
    ('freq', frequency_transformer, frequency_cols),
    ('bool', boolean_transformer, boolean_cols)
])

def preprocess_data() -> tuple[np.ndarray, pd.Series]:
    df = load_data()
    df = cleaning_engineering_pipeline.transform(df)
    X = df.drop(columns=["is_MDR"])
    y = df["is_MDR"]
    return X, y
"""
Data preprocessing utilities for the Bacteria Resistance dataset.
"""

import utils.utils as utils
import pandas as pd                                                                    # for data manipulation
import numpy as np                                                                     # for numerical operations
from pathlib import Path                                                               # for handling file paths
from sklearn.pipeline import Pipeline                                                  # for creating machine learning pipelines
from sklearn.compose import ColumnTransformer                                          # for column-wise transformations
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder   # for data preprocessing
from sklearn.impute import SimpleImputer                                               # for imputing missing values
from sklearn.base import BaseEstimator, TransformerMixin                               # for custom transformers

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
    # Optional drop columns
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors='ignore')

    return df

numerical_cols = ['infection_freq', 'age_comorb']
boolean_cols = ['ctx/cro_resistant']
categorical_cols = ['gender', 'age_bin']
frequency_cols = ['strain']

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_map = None

    def fit(self, X, y=None):
        # Different cases depending on input type
        if isinstance(X, pd.DataFrame):
            s = X['strain']
        elif isinstance(X, pd.Series):
            s = X
        elif isinstance(X, np.ndarray):
            # If 2D array with single column, convert to 1D
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
        # Returns the name of the created feature
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

def add_resistance_features(df: pd.DataFrame) -> pd.DataFrame:
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
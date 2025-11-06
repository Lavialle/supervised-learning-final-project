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
import pandas as pd # for data manipulation
import numpy as np # for numerical operations

from IPython.display import display # for displaying dataframes
from sklearn.model_selection import train_test_split, cross_val_score
from ydata_profiling import ProfileReport # for generating data profile reports
from pathlib import Path # for handling file paths

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

# Generate a profile report to understand the data and better assess the preprocessing steps needed
def generate_profile_report(df: pd.DataFrame, output_path: Path, title: str = "Profiling Report on Bacteria Dataset") -> None:
    profile = ProfileReport(
        df,
        title=title,
        explorative=True,
    )
    profile.to_file(output_path)
    print(f"Profile report generated at: {output_path}")

# Normalizing data for NaN values
def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_tokens = ['?', 'missing', 'Missing', 'None', 'none', '', 'NaN', 'nan']
    df.replace(to_replace=missing_tokens, value=np.nan, inplace=True)
    return df

# Clean the column names
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
)
    return df

# Split age/gender column into two separate columns
def split_age_gender(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[["age", "gender"]] = df["age/gender"].str.split("/", expand=True)
    df.drop(columns=["age/gender"], inplace=True)
    return df


# Split souches column to get clean strain names
def split_id_strain(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[['strain_code', 'strain']] = df['souches'].str.split(' ', n=1, expand=True) 
    df.drop(columns=["strain_code","souches"], inplace=True)
    return df 

def clean_strain_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["strain_clean"] = (
    df["strain"]
    .str.lower()
    .str.replace(r"[^a-z\s\.]", "", regex=True)  
    .str.replace(r"\s+", " ", regex=True)        
    .str.strip()                                 
)
    return df

def normalize_bacteria(name: str) -> str:

    if pd.isna(name):
        return name

    name = name.lower().strip()

    if "coli" in name or "coi" in name or "cli" in name:
        return "Escherichia coli"
    elif "enterobacter" in name or "ente" in name:
        return "Enterobacteria spp."
    elif "proteus" in name or "protus" in name or "proeus" in name or "prot" in name:
        return "Proteus mirabilis"
    elif "klebsiella" in name or "klbsiella" in name or "klebsie" in name:
        return "Klebsiella pneumoniae"
    elif "citrobacter" in name:
        return "Citrobacter spp."
    elif "morganella" in name:
        return "Morganella morganii"
    elif "serratia" in name:
        return "Serratia marcescens"
    elif "pseudomonas" in name:
        return "Pseudomonas aeruginosa"
    elif "acinetobacter" in name:
        return "Acinetobacter baumannii"
    else:
        return name.capitalize()
    
def normalize_strain_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["strain_norm"] = df["strain_clean"].apply(normalize_bacteria)
    df = df.drop(columns=["strain", "strain_clean"])
    return df

# Uniformize antibiotic susceptibility values
def norm_ast(v):
    if pd.isna(v): 
        return np.nan
    s = str(v).strip().upper()
    if s=='' or s in {'NA','N/A','?','MISSING'}: 
        return np.nan
    if s.startswith('R'): 
        return 'R'
    if s.startswith('S'): 
        return 'S'
    if s.startswith('I') or 'INTER' in s: 
        return 'I'
    return np.nan

def uniformize_susceptibility_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ab_cols = [
        'amx/amp', 'amc', 'cz', 'fox', 'ctx/cro', 'ipm', 'gen',
        'an', 'acide_nalidixique', 'ofx', 'cip', 'c',
        'co-trimoxazole', 'furanes', 'colistine'
    ]
    for col in ab_cols:
        df[col+"_norm"] = df[col].apply(norm_ast)
        df.drop(columns=[col], inplace=True)
    return df

# Remove duplicate rows
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop_duplicates(inplace=True)
    return df

# Normalize boolean columns
def normalize_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bool_map = {'No': False, 'Yes': True, 'True': True}

    bool_cols = ['diabetes', 'hypertension', 'hospital_before']

    for col in bool_cols:
        df[col] = df[col].map(bool_map)
    return df

# Clean and parse to datetime the collection_date column

# Create a function to parse different formats
def parse_date(val):
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y"):
        try:
            return pd.to_datetime(val, format=fmt, dayfirst=True)
        except:
            continue
    return pd.NaT

def clean_collection_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Replace month abbreviations 
    month_map = {
        'Fev': 'Feb',  
    }
    for k, v in month_map.items():
        df['collection_date'] = df['collection_date'].str.replace(k, v, regex=False)

    # Apply parsing function
    df['collection_date'] = df['collection_date'].apply(lambda x: parse_date(str(x)))

    # Format as dd-mm-yyyy string
    df['collection_date'] = df['collection_date'].dt.strftime('%d-%m-%Y')

    df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce', dayfirst=True)
    return df

def handle_missing_dates(df: pd.DataFrame, date_col: str = "collection_date") -> pd.DataFrame:
    """
    Nettoie et impute les dates manquantes dans une colonne de type date.
    - Convertit la colonne en datetime.
    - Remplace les valeurs manquantes par la médiane des dates existantes.
    """
    df = df.copy()

    if date_col not in df.columns:
        return df  # si la colonne n'existe pas, on ne fait rien

    # Conversion en datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    # Si toutes les valeurs sont manquantes, on laisse la colonne telle quelle
    if df[date_col].isna().all():
        print(f"Toutes les dates sont manquantes dans '{date_col}', aucune imputation faite.")
        return df

    # Calcul de la médiane (par ex. mi-période)
    median_date = df[date_col].dropna().median()

    # Remplissage des NaN
    df[date_col] = df[date_col].fillna(median_date)

    return df


# Drop rows where all columns EXCEPT ['id', 'address','collection_date','notes'] are NaN
def drop_all_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    subset_cols = df.columns.difference(['id','name', 'email', 'address', 'collection_date', 'notes'])
    df.dropna(how='all', subset=subset_cols, inplace=True)
    return df

# Return the list of the numerical, boolean and categorical columns
def get_column_types(df: pd.DataFrame):
    numerical_cols = ['age', 'infection_freq']
    boolean_cols = ['diabetes', 'hypertension', 'hospital_before']
    categorical_cols = [
        "gender",'amx/amp_norm', 'amc_norm', 'cz_norm', 'fox_norm', 'ctx/cro_norm', 'ipm_norm', 'gen_norm',
        'an_norm', 'acide_nalidixique_norm', 'ofx_norm', 'cip_norm', 'c_norm',
        'co-trimoxazole_norm', 'furanes_norm', 'colistine_norm', 'strain_norm'
    ]
    date_cols = ['collection_date']
    return numerical_cols, boolean_cols, categorical_cols, date_cols

# Cast categorical columns to 'category' dtype
def cast_categorical_columns(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy() 
    cat_cols = get_column_types(df)[2] 
    df[cat_cols] = df[cat_cols].astype('category') 
    return df 

# Cast boolean columns to 'boolean' dtype 
def cast_boolean_columns(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy() 
    bool_cols = get_column_types(df)[1] 
    df[bool_cols] = df[bool_cols].astype('boolean') 
    return df 
# Impute numerical columns using IterativeImputer # 
def cast_numerical_columns(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy() 
    num_cols = get_column_types(df)[0]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    return df

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
    df = clean_column_names(df)
    df = normalize_missing_values(df)
    df = split_age_gender(df)
    df = split_id_strain(df)
    df = clean_strain_names(df)
    df = normalize_strain_names(df)
    df = uniformize_susceptibility_values(df)
    df = drop_duplicates(df)
    df = normalize_boolean_columns(df)
    df = clean_collection_date(df)
    df = handle_missing_dates(df, date_col="collection_date")  # imputation dates

    # Empty rows removal
    df = drop_all_nan_rows(df)

    # Cast columns
    df = cast_boolean_columns(df)
    df = cast_categorical_columns(df)
    df = cast_numerical_columns(df)

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
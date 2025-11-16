"""
Utility functions for data preprocessing and feature engineering.
"""

import pandas as pd # for data manipulation
import numpy as np # for numerical operations
# from ydata_profiling import ProfileReport # for generating data profile reports
# from pathlib import Path # for handling file paths

# # Generate a profile report to understand the data and better assess the preprocessing steps needed
# def generate_profile_report(df: pd.DataFrame, output_path: Path, title: str = "Profiling Report on Bacteria Dataset") -> None:
#     profile = ProfileReport(
#         df,
#         title=title,
#         explorative=True,
#     )
#     profile.to_file(output_path)
#     print(f"Profile report generated at: {output_path}")

# ====================================================================================================================================
# DATA CLEANING
# ====================================================================================================================================

#  Normalizing data for NaN values
def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_tokens = ['?', 'missing', 'Missing', 'None', 'none', '', 'NaN', 'nan', 'unknown', 'error', 'n/a', '(Missing)']
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
    df["strain"] = df["strain_clean"].apply(normalize_bacteria)
    df = df.drop(columns=["strain_clean"])
    return df

# Uniformize antibiotic susceptibility values
def norm_ast(v):
    if pd.isna(v): 
        return np.nan
    s = str(v).strip().upper()
    if s =='' or s in {'NA','N/A','?','MISSING'}: 
        return np.nan
    if s.startswith('R'): 
        return 'R'
    if s.startswith('S'): 
        return 'S'
    if s.startswith('I') or 'INTER' in s: 
        return 'S'
    return np.nan

def uniformize_susceptibility_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ab_cols = [
        'amx/amp', 'amc', 'cz', 'fox', 'ctx/cro', 'ipm', 'gen',
        'an', 'acide_nalidixique', 'ofx', 'cip', 'c',
        'co-trimoxazole', 'furanes', 'colistine'
    ]
    for col in ab_cols:
        # Appply normalization
        df[col] = df[col].apply(norm_ast)
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
def parse_date(val: str) -> pd.Timestamp:
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

# Drop rows where all columns EXCEPT ['id', 'address','collection_date','notes'] are NaN
def drop_all_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    subset_cols = df.columns.difference(['id','name', 'email', 'address', 'collection_date', 'notes'])
    df.dropna(how='all', subset=subset_cols, inplace=True)
    return df



# ==================================================================================================================================
# DATA ENGINEERING
# ==================================================================================================================================

def compute_antibiotic_resistance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ab_cols = [
        'amx/amp', 'amc', 'cz', 'fox', 'ctx/cro', 'ipm', 'gen',
        'an', 'acide_nalidixique', 'ofx', 'cip', 'c',
        'co-trimoxazole', 'furanes', 'colistine'
    ]

    for col in ab_cols:
        df[col + '_resistant'] = (df[col] == 'R').astype(int)
    df.drop(columns=ab_cols, inplace=True)
    return df
    

def compute_family_resistance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    families = {
        'beta_lactams': ['amx/amp', 'amc', 'cz', 'fox', 'ctx/cro', 'ipm'],
        'aminosides': ['gen', 'an'],
        'quinolones': ['acide_nalidixique', 'ofx', 'cip'],
        'phenicols': ['c'],
        'sulfamides': ['co-trimoxazole'],
        'nitrofuranes': ['furanes'],
        'polymyxines': ['colistine']
    }

   # Count resistant families
    def count_resistant_families(row: pd.Series) -> int:
        resistant_families = 0
        for family, cols in families.items():
            for col in cols:
                val = str(row.get(col, ""))
                if val == "R":
                    resistant_families += 1
                    break
        return resistant_families

    df["n_resistant_families"] = df.apply(count_resistant_families, axis=1)

    # Classify MDR/XDR/PDR
    def classify_resistance(row: pd.Series) -> pd.DataFrame:
        if row['n_resistant_families'] >= 3:
            return 'MDR'
        else:
            return 'Non-MDR'

    df['resistance_profile'] = df.apply(classify_resistance, axis=1)
    return df

def compute_is_MDR(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['is_MDR'] = df['resistance_profile'].apply(lambda x: 1 if x == 'MDR' else 0)
    df.drop(columns=['resistance_profile', 'n_resistant_families'], inplace=True)
    return df

def add_age_comorbidity_interaction(df: pd.DataFrame, age_col="age", comorb_cols=None) -> pd.DataFrame:
    df = df.copy()
    if comorb_cols is None:
        comorb_cols = ["diabetes", "hypertension", "hospital_before"]
    df["comorbidity_score"] = df[comorb_cols].sum(axis=1)
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df["age_comorb"] = df[age_col] * df["comorbidity_score"]
    return df

def bin_age_and_drop(df: pd.DataFrame, age_col="age", bins=None, labels=None) -> pd.DataFrame:
    df = df.copy()
    if bins is None:
        bins = [0, 18, 40, 65, 100]
    if labels is None:
        labels = ["child", "adult", "senior", "elderly"]
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df["age_bin"] = pd.cut(df[age_col], bins=bins, labels=labels, include_lowest=True)
    return df

# Return the list of the numerical, boolean and categorical columns
def get_column_types(df: pd.DataFrame):
    """
    Return (numerical_cols, boolean_cols, categorical_cols) — only existing columns.
    """
    cols = set(df.columns)

    numerical_candidates = ['infection_freq', 'age_comorb', 'comorbidity_score', 'age']
    numerical_cols = [c for c in numerical_candidates if c in cols]

    boolean_cols = [c for c in df.columns if c.endswith('_resistant') or c in {'diabetes', 'hypertension', 'hospital_before', 'is_MDR'}]
    # keep order and unique
    boolean_cols = list(dict.fromkeys(boolean_cols))

    categorical_candidates = ['gender', 'strain', 'age_bin']
    categorical_cols = [c for c in categorical_candidates if c in cols]

    return numerical_cols, boolean_cols, categorical_cols

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

# Cast numerical columns to 'float' dtype 
def cast_numerical_columns(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy() 
    num_cols = get_column_types(df)[0]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    return df

def drop_correlated_features(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.drop(columns=[
        'amx/amp_resistant', 'amc_resistant', 'cz_resistant', 'fox_resistant', 'ipm_resistant',
        'gen_resistant', 'an_resistant', 'acide_nalidixique_resistant', 'ofx_resistant', 'cip_resistant', 'c_resistant',
        'co-trimoxazole_resistant', 'furanes_resistant', 'colistine_resistant', 'comorbidity_score', 'hospital_before',
        'hypertension', 'diabetes', 'age'
    ])
    return df

def drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    subset_cols = df.columns.difference(['is_MDR'])
    df.dropna(how='all', subset=subset_cols, inplace=True)
    return df
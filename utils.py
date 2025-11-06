import pandas as pd # for data manipulation
import numpy as np # for numerical operations
from ydata_profiling import ProfileReport # for generating data profile reports
from pathlib import Path # for handling file paths

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
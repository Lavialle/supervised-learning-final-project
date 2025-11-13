"""This module is going to have several objectives.
1. Load the dataset from a CSV file.
2. Preprocess the data using Pipeline module from sklearn.
3. Train a machine learning model.
4. Evaluate the model's performance.
"""

# 0. Get the requirements from requirements.txt and install them using pip if not already installed.


# Import necessary librairies
import utils  # custom utility functions for data preprocessing
import mlflow
import mlflow.sklearn
import pandas as pd # for data manipulation
import numpy as np # for numerical operations
import matplotlib.pyplot as plt
from pathlib import Path # for handling file paths
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import StackingClassifier
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

# Optimization imports
from hyperopt import fmin, tpe, hp


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
numerical_cols = ['infection_freq', 'age_comorb']
boolean_cols = ['age_bin_child', 'age_bin_adult', 'age_bin_senior', 'age_bin_elderly']#['diabetes', 'hypertension', 'hospital_before']
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
    ('RandomForest', RandomForestClassifier(random_state=42, class_weight='balanced'))
])


CatBoost = Pipeline([
    ('preprocess', preprocessing),
    ('CatBoost', CatBoostClassifier(verbose=0, scale_pos_weight=4510/2304, random_state=42))
]) 

XGBoost = Pipeline([
    ('preprocess', preprocessing),
    ('XGBoost', XGBClassifier(scale_pos_weight=4510/2304, random_state=42))
])

Logistic = Pipeline([
    ('preprocess', preprocessing),
    ('Logistic', LogisticRegression(class_weight='balanced'))        
])

HistGradient = Pipeline([
    ('preprocess', preprocessing),
    ('HistGradient', HistGradientBoostingClassifier(random_state=42, class_weight='balanced'))    
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
print(X.info())
print(X.isna().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "CatBoost": CatBoost,
    "XGBoost": XGBoost,
    "HistGradient": HistGradient
}

search_spaces = {
    "CatBoost": {
        "CatBoost__depth": hp.quniform("CatBoost__depth", 3, 10, 1),
        "CatBoost__learning_rate": hp.uniform("CatBoost__learning_rate", 0.01, 0.3),
        "CatBoost__iterations": hp.quniform("CatBoost__iterations", 100, 500, 50),
    },
    "XGBoost": {
        "XGBoost__max_depth": hp.quniform("XGBoost__max_depth", 3, 10, 1),
        "XGBoost__n_estimators": hp.quniform("XGBoost__n_estimators", 100, 500, 50),
        "XGBoost__learning_rate": hp.uniform("XGBoost__learning_rate", 0.01, 0.3),
    },
    "HistGradient": {
        "HistGradient__max_depth": hp.quniform("HistGradient__max_depth", 3, 10, 1),
        "HistGradient__max_iter": hp.quniform("HistGradient__max_iter", 100, 500, 50),
        "HistGradient__learning_rate": hp.uniform("HistGradient__learning_rate", 0.01, 0.3),
    }
}

stack_space = {
    'catboost__depth': hp.quniform('catboost__depth', 4, 8, 1),
    'catboost__learning_rate': hp.uniform('catboost__learning_rate', 0.01, 0.2),
    'xgb__max_depth': hp.quniform('xgb__max_depth', 4, 8, 1),
    'xgb__learning_rate': hp.uniform('xgb__learning_rate', 0.01, 0.2),
    'lgbm__num_leaves': hp.quniform('lgbm__num_leaves', 20, 40, 1),
    'final_estimator__C': hp.loguniform('final_estimator__C', -3, 1)
}

# best = {}

# for name, model in models.items():
#     print(f"\nOptimisation pour {name}...")

#     # Évaluation simple avant optimisation
#     scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
#     print(f"{name} F1: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

#     # Fonction objectif pour hyperopt
#     def objective(params):
#         for key in params:
#             if "depth" in key or "iterations" in key or "n_estimators" in key or "max_iter" in key:
#                 params[key] = int(params[key])
#         model.set_params(**params)
#         score = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()
#         return -score  # car fmin minimise


#     best_params = fmin(
#         fn=objective,
#         space=search_spaces[name],
#         algo=tpe.suggest,
#         max_evals=20,
#     )

#     best[name] = best_params
#     print(f"Best params for {name}: {best_params}")

# print("\nRésultats finaux :")
# print(best)


# for name, model in models.items():
#     # Convertir les hyperopt float -> int si nécessaire
#     params = {k: int(v) if k.endswith("depth") or k.endswith("iterations") or k.endswith("n_estimators") or k.endswith("max_iter") else v
#               for k,v in best[name].items()}
#     model.set_params(**params)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(f"--- {name} ---")
#     print(classification_report(y_test, y_pred))
#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-MDR', 'MDR'])
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title(f"Confusion Matrix - {name}")
#     plt.show()

def to_dataframe(X):
    # Si X est déjà un DataFrame, retourne-le
    if isinstance(X, pd.DataFrame):
        return X
    # Sinon, crée un DataFrame avec des noms génériques
    return pd.DataFrame(X)

def stacking_objective(params):
    cat_params = {
        'depth': int(params['catboost__depth']),
        'learning_rate': params['catboost__learning_rate'],
        'iterations': int(params['catboost__iterations']),
        'l2_leaf_reg': params['catboost__l2_leaf_reg'],
        'verbose': 0,
        'scale_pos_weight': 4510/2304,
        'random_state': 42
    }
    xgb_params = {
        'max_depth': int(params['xgb__max_depth']),
        'learning_rate': params['xgb__learning_rate'],
        'n_estimators': int(params['xgb__n_estimators']),
        'subsample': params['xgb__subsample'],
        'colsample_bytree': params['xgb__colsample_bytree'],
        'gamma': params['xgb__gamma'],
        'scale_pos_weight': 4510/2304,
        'random_state': 42
    }
    lgbm_params = {
        'num_leaves': int(params['lgbm__num_leaves']),
        'min_child_samples': int(params['lgbm__min_child_samples']),
        'subsample': params['lgbm__subsample'],
        'reg_alpha': params['lgbm__reg_alpha'],
        'reg_lambda': params['lgbm__reg_lambda'],
        'random_state': 42,
        'verbose': -1
    }
    final_params = {
        'n_estimators': int(params['rf__n_estimators']),
        'max_depth': int(params['rf__max_depth']),
        'class_weight': 'balanced',
        'random_state': 42
    }

    cat = Pipeline([
        ('preprocess', preprocessing),
        ('catboost', CatBoostClassifier(**cat_params))
    ])
    xgb = Pipeline([
        ('preprocess', preprocessing),
        ('xgb', XGBClassifier(**xgb_params))
    ])
    lgbm = Pipeline([
        ('preprocess', preprocessing),
        ('to_df', FunctionTransformer(to_dataframe, validate=False)),
        ('lgbm', LGBMClassifier(**lgbm_params))
    ])
    final_est = RandomForestClassifier(**final_params)

    stack = StackingClassifier(
        estimators=[
            ('cat', cat),
            ('xgb', xgb),
            ('lgbm', lgbm)
        ],
        final_estimator=final_est,
        cv=cv,
        n_jobs=-1
    )

    score = cross_val_score(stack, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()
    return -score

stack_space = {
    'catboost__depth': hp.quniform('catboost__depth', 4, 8, 1),
    'catboost__learning_rate': hp.uniform('catboost__learning_rate', 0.01, 0.2),
    'catboost__iterations': hp.quniform('catboost__iterations', 100, 400, 50),
    'catboost__l2_leaf_reg': hp.uniform('catboost__l2_leaf_reg', 1, 10),
    'xgb__max_depth': hp.quniform('xgb__max_depth', 4, 8, 1),
    'xgb__learning_rate': hp.uniform('xgb__learning_rate', 0.01, 0.2),
    'xgb__n_estimators': hp.quniform('xgb__n_estimators', 100, 400, 50),
    'xgb__subsample': hp.uniform('xgb__subsample', 0.6, 1.0),
    'xgb__colsample_bytree': hp.uniform('xgb__colsample_bytree', 0.6, 1.0),
    'xgb__gamma': hp.uniform('xgb__gamma', 0, 5),
    'lgbm__num_leaves': hp.quniform('lgbm__num_leaves', 20, 40, 1),
    'lgbm__min_child_samples': hp.quniform('lgbm__min_child_samples', 10, 30, 1),
    'lgbm__subsample': hp.uniform('lgbm__subsample', 0.6, 1.0),
    'lgbm__reg_alpha': hp.uniform('lgbm__reg_alpha', 0, 2),
    'lgbm__reg_lambda': hp.uniform('lgbm__reg_lambda', 0, 2),
    'rf__n_estimators': hp.quniform('rf__n_estimators', 100, 400, 50),
    'rf__max_depth': hp.quniform('rf__max_depth', 4, 10, 1)
}

print("\nOptimisation du modèle stacké avec RandomForest méta-modèle...")
best_stack = fmin(
    fn=stacking_objective,
    space=stack_space,
    algo=tpe.suggest,
    max_evals=10,
)

print("Best params for stacking:", best_stack)

# Fit final stacking model with best params
cat_params = {
    'depth': int(best_stack['catboost__depth']),
    'learning_rate': best_stack['catboost__learning_rate'],
    'iterations': int(best_stack['catboost__iterations']),
    'l2_leaf_reg': best_stack['catboost__l2_leaf_reg'],
    'verbose': 0,
    'scale_pos_weight': 4510/2304,
    'random_state': 42
}
xgb_params = {
    'max_depth': int(best_stack['xgb__max_depth']),
    'learning_rate': best_stack['xgb__learning_rate'],
    'n_estimators': int(best_stack['xgb__n_estimators']),
    'subsample': best_stack['xgb__subsample'],
    'colsample_bytree': best_stack['xgb__colsample_bytree'],
    'gamma': best_stack['xgb__gamma'],
    'scale_pos_weight': 4510/2304,
    'random_state': 42
}
lgbm_params = {
    'num_leaves': int(best_stack['lgbm__num_leaves']),
    'min_child_samples': int(best_stack['lgbm__min_child_samples']),
    'subsample': best_stack['lgbm__subsample'],
    'reg_alpha': best_stack['lgbm__reg_alpha'],
    'reg_lambda': best_stack['lgbm__reg_lambda'],
    'random_state': 42,
    'verbose': -1
}
final_params = {
    'n_estimators': int(best_stack['rf__n_estimators']),
    'max_depth': int(best_stack['rf__max_depth']),
    'class_weight': 'balanced',
    'random_state': 42
}

cat = Pipeline([
    ('preprocess', preprocessing),
    ('catboost', CatBoostClassifier(**cat_params))
])

xgb = Pipeline([
    ('preprocess', preprocessing),
    ('xgb', XGBClassifier(**xgb_params))
])

lgbm = Pipeline([
    ('preprocess', preprocessing),
    ('to_df', FunctionTransformer(to_dataframe, validate=False)),
    ('lgbm', LGBMClassifier(**lgbm_params))
])

final_est = RandomForestClassifier(**final_params)

stack = StackingClassifier(
    estimators=[
        ('cat', cat),
        ('xgb', xgb),
        ('lgbm', lgbm)
    ],
    final_estimator=final_est,
    cv=cv,
    n_jobs=-1
)

stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)
print("--- Stacking (RandomForest méta-modèle) ---")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-MDR', 'MDR'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Stacking (RF meta)")
plt.show()



# RandomForest.fit(X_train, y_train)
# rf_pred = RandomForest.predict(X_test)
# cm_rf = confusion_matrix(y_test, rf_pred)
# print(cm_rf)
# print(classification_report(y_test, rf_pred))

# CatBoost.fit(X_train, y_train)
# cb_pred = CatBoost.predict(X_test)
# cm_cb = confusion_matrix(y_test, cb_pred)
# print(cm_cb)
# print(classification_report(y_test, cb_pred))

# XGBoost.fit(X_train, y_train)
# xg_pred = XGBoost.predict(X_test)
# cm_xg = confusion_matrix(y_test, xg_pred)
# print(cm_xg)
# print(classification_report(y_test, xg_pred))


# Hyperparameters optimization












# param_grid = {
#     "HistGradient__learning_rate": [0.01, 0.05, 0.1],
#     "HistGradient__max_iter": [100, 200],
#     "HistGradient__max_depth": [3, 5, None]
# }

# grid = GridSearchCV(HistGradient, param_grid, scoring='f1', cv=5, n_jobs=-1)





# # MLflow
# mlflow.set_tracking_uri("http://127.0.0.1:8080")
# mlflow.set_experiment("bacteria_resistance_classification")

# with mlflow.start_run() as run:
#     mlflow.set_tag("model_type", "HistGradientBoostingClassifier")

#     # Train
#     grid.fit(X_train, y_train)

#     # Best model
#     best_model = grid.best_estimator_
#     mlflow.log_params(grid.best_params_)

#     # Predict
#     y_pred = best_model.predict(X_test)
#     f1 = f1_score(y_test, y_pred)
#     mlflow.log_metric("f1_score", f1)

#     cm = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix:\n", cm)

#     report = classification_report(y_test, y_pred)
#     print(report)

#     # Optionally log the report as text
#     mlflow.log_text(report, "classification_report.txt")

#     # Log the model
#     mlflow.sklearn.log_model(best_model, artifact_path="model")

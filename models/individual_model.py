"""
Individual model training and evaluation.
Models included: RandomForest, CatBoost, XGBoost, Logistic Regression.
"""

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils.preprocessing import preprocess_data, preprocessing
from utils.evaluation import evaluate_model

def run_individual():
    X, y = preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    RandomForest = Pipeline([
        ('preprocess', preprocessing),
        ('RandomForest', RandomForestClassifier(random_state=42))
    ])

    CatBoost = Pipeline([
        ('preprocess', preprocessing),
        ('CatBoost', CatBoostClassifier(verbose=0, random_state=42))
    ]) 

    XGBoost = Pipeline([
        ('preprocess', preprocessing),
        ('XGBoost', XGBClassifier(random_state=42))
    ])

    Logistic = Pipeline([
        ('preprocess', preprocessing),
        ('Logistic', LogisticRegression(random_state=42))        
    ])

    models = {
        "RandomForest": RandomForest,
        "CatBoost": CatBoost,
        "XGBoost": XGBoost,
        "Logistic": Logistic
    }

    for model_name, model in models.items():

        print(f"Training and evaluating {model_name}...")
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        print(f"{model_name} CV F1 scores:", score)
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test, model_name)


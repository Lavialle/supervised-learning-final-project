"""
Hyperopt optimization for CatBoost and XGBoost models.
"""

from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from utils.preprocessing import preprocess_data, preprocessing
from utils.evaluation import evaluate_model
import numpy as np
from hyperopt import fmin, tpe, hp, Trials

def run_hyperopt():
    # Prétraitement des données
    X, y = preprocess_data()
    proportion_MDR = (y == 0).sum() / (y == 1).sum()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    CatBoost = Pipeline([
        ('preprocess', preprocessing),
        ('CatBoost', CatBoostClassifier(verbose=0, random_state=42))
    ])

    XGBoost = Pipeline([
        ('preprocess', preprocessing),
        ('XGBoost', XGBClassifier(random_state=42))
    ])

    models = {
        "CatBoost": CatBoost,
        "XGBoost": XGBoost,
    }

    # Définir les espaces de recherche pour Hyperopt
    search_spaces = {
        "CatBoost": {
            'CatBoost__depth': hp.quniform('CatBoost__depth', 4, 8, 1),
            'CatBoost__learning_rate': hp.uniform('CatBoost__learning_rate', 0.01, 0.2),
            'CatBoost__iterations': hp.quniform('CatBoost__iterations', 200, 600, 50),
            'CatBoost__l2_leaf_reg': hp.uniform('CatBoost__l2_leaf_reg', 1, 10),
            'CatBoost__scale_pos_weight': proportion_MDR,
            'CatBoost__random_state': 42
        },
        "XGBoost": {
            'XGBoost__max_depth': hp.quniform('XGBoost__max_depth', 4, 10, 1),
            'XGBoost__learning_rate': hp.uniform('XGBoost__learning_rate', 0.01, 0.2),
            'XGBoost__n_estimators': hp.quniform('XGBoost__n_estimators', 200, 600, 50),
            'XGBoost__subsample': hp.uniform('XGBoost__subsample', 0.6, 1.0),
            'XGBoost__colsample_bytree': hp.uniform('XGBoost__colsample_bytree', 0.6, 1.0),
            'XGBoost__gamma': hp.uniform('XGBoost__gamma', 0, 5),
            'XGBoost__scale_pos_weight': proportion_MDR,
            'XGBoost__random_state': 42,
            'XGBoost__eval_metric': 'logloss'
        }
    }

    best = {}

    for name, model in models.items():
        print(f"\nOptimisation pour {name}...")

        # Évaluation simple avant optimisation
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

        # Fonction objectif pour Hyperopt
        def objective(params):
            # Convertir les hyperparamètres en entiers si nécessaire
            for key in params:
                if "depth" in key or "iterations" in key or "n_estimators" in key:
                    params[key] = int(params[key])
            model.set_params(**params)
            score = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()
            return -score  # car fmin minimise

        # Optimisation avec Hyperopt
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=search_spaces[name],
            algo=tpe.suggest,
            max_evals=20,
            trials=trials
        )

        # Convertir les hyperparamètres optimaux en entiers si nécessaire
        best_params = {k: int(v) if k.endswith("depth") or k.endswith("iterations") or k.endswith("n_estimators") else v
                       for k, v in best_params.items()}

        best[name] = best_params

        # Entraîner le modèle avec les meilleurs hyperparamètres
        model.set_params(**best_params)
        model.fit(X_train, y_train)

        # Évaluer le modèle optimisé
        evaluate_model(model, X_test, y_test, name)


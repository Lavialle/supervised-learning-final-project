"""
Stacking model with Hyperopt optimization.
Models used: CatBoost, XGBoost, LightGBM as base models and RandomForest as meta-model.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from hyperopt import fmin, tpe, hp, Trials
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocessing import preprocess_data, preprocessing
from utils.evaluation import evaluate_model



def run_stacking():
    X, y = preprocess_data()
    proportion_MDR = (y == 0).sum() / (y == 1).sum()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # LIGHTGBM needs a DataFrame input after preprocessing
    def to_dataframe(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    # Objective function for Hyperopt optimization of the stacking model
    def stacking_objective(params):
        # --- CATBOOST
        cat_params = {
            'depth': int(params['CatBoost__depth']),
            'learning_rate': params['CatBoost__learning_rate'],
            'iterations': int(params['CatBoost__iterations']),
            'l2_leaf_reg': params['CatBoost__l2_leaf_reg'],
            'verbose': 0,
            'scale_pos_weight': proportion_MDR,
            'random_state': 42
        }
        # --- XGBOOST
        xgb_params = {
            'max_depth': int(params['XGBoost__max_depth']),
            'learning_rate': params['XGBoost__learning_rate'],
            'n_estimators': int(params['XGBoost__n_estimators']),
            'subsample': params['XGBoost__subsample'],
            'colsample_bytree': params['XGBoost__colsample_bytree'],
            'gamma': params['XGBoost__gamma'],
            'scale_pos_weight': proportion_MDR,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        # --- LIGHTGBM
        lgbm_params = {
            'num_leaves': int(params['LightGBM__num_leaves']),
            'min_child_samples': int(params['LightGBM__min_child_samples']),
            'subsample': params['LightGBM__subsample'],
            'reg_alpha': params['LightGBM__reg_alpha'],
            'reg_lambda': params['LightGBM__reg_lambda'],
            'random_state': 42,
            'verbose': -1
        }
        logreg_params = {
            'C': params['LogisticRegression__C'],
            'max_iter': int(params['LogisticRegression__max_iter']),
            'class_weight': 'balanced',
            'random_state': 42
        }
        # --- META RandomForest
        final_params = {
            'n_estimators': int(params['RandomForest__n_estimators']),
            'max_depth': int(params['RandomForest__max_depth']),
            'class_weight': 'balanced',
            'random_state': 42
        }
        # Define base learners with current hyperparameters
        CatBoost = Pipeline([
            ('preprocess', preprocessing),
            ('CatBoost', CatBoostClassifier(**cat_params))
        ])
        XGBoost = Pipeline([
            ('preprocess', preprocessing),
            ('XGBoost', XGBClassifier(**xgb_params))
        ])
        LightGBM = Pipeline([
            ('preprocess', preprocessing),
            ('to_df', FunctionTransformer(to_dataframe, validate=False)),
            ('LightGBM', LGBMClassifier(**lgbm_params))
        ])
        LogisticReg = Pipeline([
            ('preprocess', preprocessing),
            ('LogisticRegression', LogisticRegression(**logreg_params))
        ])

        # Define meta-learner
        final_est = RandomForestClassifier(**final_params)

        stack = StackingClassifier(
            estimators=[
                ('CatBoost', CatBoost),
                ('XGBoost', XGBoost),
                ('LightGBM', LightGBM),
                ('LogisticRegression', LogisticReg)
            ],
            final_estimator=final_est,
            cv=cv,
            n_jobs=-1
        )

        # Average F1-score on CV
        score = cross_val_score(stack, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()
        return -score  # because fmin minimizes

    # Define the search space for Hyperopt
    stack_space = {
        # --- CatBoost
        'CatBoost__depth': hp.quniform('CatBoost__depth', 4, 10, 1),
        'CatBoost__learning_rate': hp.loguniform('CatBoost__learning_rate', np.log(0.01), np.log(0.3)), 
        'CatBoost__iterations': hp.quniform('CatBoost__iterations', 100, 800, 50),
        'CatBoost__l2_leaf_reg': hp.uniform('CatBoost__l2_leaf_reg', 1, 10),
        'CatBoost__border_count': hp.quniform('CatBoost__border_count', 32, 255, 1), 
        # --- XGBoost
        'XGBoost__max_depth': hp.quniform('XGBoost__max_depth', 3, 12, 1),  
        'XGBoost__learning_rate': hp.loguniform('XGBoost__learning_rate', np.log(0.01), np.log(0.3)),
        'XGBoost__n_estimators': hp.quniform('XGBoost__n_estimators', 100, 800, 50),
        'XGBoost__subsample': hp.uniform('XGBoost__subsample', 0.6, 1.0),
        'XGBoost__colsample_bytree': hp.uniform('XGBoost__colsample_bytree', 0.6, 1.0),
        'XGBoost__gamma': hp.uniform('XGBoost__gamma', 0, 5),
        'XGBoost__reg_alpha': hp.uniform('XGBoost__reg_alpha', 0, 2),  
        'XGBoost__reg_lambda': hp.uniform('XGBoost__reg_lambda', 0, 2),  
        'XGBoost__min_child_weight': hp.quniform('XGBoost__min_child_weight', 1, 10, 1),  
        # --- LightGBM
        'LightGBM__num_leaves': hp.quniform('LightGBM__num_leaves', 20, 100, 5),
        'LightGBM__learning_rate': hp.loguniform('LightGBM__learning_rate', np.log(0.01), np.log(0.3)),
        'LightGBM__n_estimators': hp.quniform('LightGBM__n_estimators', 100, 800, 50),
        'LightGBM__max_depth': hp.quniform('LightGBM__max_depth', 3, 12, 1), 
        'LightGBM__min_child_samples': hp.quniform('LightGBM__min_child_samples', 5, 50, 5),
        'LightGBM__subsample': hp.uniform('LightGBM__subsample', 0.6, 1.0),
        'LightGBM__colsample_bytree': hp.uniform('LightGBM__colsample_bytree', 0.6, 1.0),
        'LightGBM__reg_alpha': hp.uniform('LightGBM__reg_alpha', 0, 2),
        'LightGBM__reg_lambda': hp.uniform('LightGBM__reg_lambda', 0, 2),
        # --- Logistic Regression
        'LogisticRegression__C': hp.loguniform('LogisticRegression__C', np.log(0.01), np.log(10)),
        'LogisticRegression__max_iter': hp.quniform('LogisticRegression__max_iter', 500, 1500, 100),
        # --- RandomForest (meta)
        'RandomForest__n_estimators': hp.quniform('RandomForest__n_estimators', 100, 500, 50),
        'RandomForest__max_depth': hp.quniform('RandomForest__max_depth', 5, 20, 1),  
        'RandomForest__min_samples_split': hp.quniform('RandomForest__min_samples_split', 2, 20, 2), 
        'RandomForest__min_samples_leaf': hp.quniform('RandomForest__min_samples_leaf', 1, 10, 1), 
    }

    print("\n==== Stacked model optimization ====")
    best_stack = fmin(
        fn=stacking_objective,
        space=stack_space,
        algo=tpe.suggest,
        max_evals=100, 
        rstate=np.random.default_rng(42),
        trials = Trials()
    )

    # Convert best params
    def param_int(v):
        try:
            # If the value is an integer (e.g. 5.0), convert it to int
            return int(v) if float(v).is_integer() else float(v)
        except:
            return v

    best_stack = {k: param_int(v) for k, v in best_stack.items()}

    # Redefine models with the best hyperparameters
    cat_params = {
        'depth': best_stack['CatBoost__depth'],
        'learning_rate': best_stack['CatBoost__learning_rate'],
        'iterations': best_stack['CatBoost__iterations'],
        'l2_leaf_reg': best_stack['CatBoost__l2_leaf_reg'],
        'verbose': 0,
        'border_count': best_stack['CatBoost__border_count'],
        'scale_pos_weight': proportion_MDR,
        'random_state': 42
    }
    xgb_params = {
        'max_depth': best_stack['XGBoost__max_depth'],
        'learning_rate': best_stack['XGBoost__learning_rate'],
        'n_estimators': best_stack['XGBoost__n_estimators'],
        'subsample': best_stack['XGBoost__subsample'],
        'colsample_bytree': best_stack['XGBoost__colsample_bytree'],
        'gamma': best_stack['XGBoost__gamma'],
        'reg_alpha': best_stack['XGBoost__reg_alpha'],
        'reg_lambda': best_stack['XGBoost__reg_lambda'],
        'min_child_weight': best_stack['XGBoost__min_child_weight'],
        'scale_pos_weight': proportion_MDR,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    lgbm_params = {
        'num_leaves': best_stack['LightGBM__num_leaves'],
        'min_child_samples': best_stack['LightGBM__min_child_samples'],
        'subsample': best_stack['LightGBM__subsample'],
        'reg_alpha': best_stack['LightGBM__reg_alpha'],
        'reg_lambda': best_stack['LightGBM__reg_lambda'],
        'colsample_bytree': best_stack['LightGBM__colsample_bytree'],
        'learning_rate': best_stack['LightGBM__learning_rate'],
        'n_estimators': best_stack['LightGBM__n_estimators'],
        'max_depth': best_stack['LightGBM__max_depth'],
        'random_state': 42,
        'verbose': -1
    }
    logreg_params = {
        'C': best_stack['LogisticRegression__C'],
        'max_iter': best_stack['LogisticRegression__max_iter'],
        'class_weight': 'balanced',
        'random_state': 42
    }
    final_params = {
        'n_estimators': best_stack['RandomForest__n_estimators'],
        'max_depth': best_stack['RandomForest__max_depth'],
        'min_samples_split': best_stack['RandomForest__min_samples_split'],
        'min_samples_leaf': best_stack['RandomForest__min_samples_leaf'],
        'class_weight': 'balanced',
        'random_state': 42
    }

    CatBoost = Pipeline([
        ('preprocess', preprocessing),
        ('CatBoost', CatBoostClassifier(**cat_params))
    ])
    XGBoost = Pipeline([
        ('preprocess', preprocessing),
        ('XGBoost', XGBClassifier(**xgb_params))
    ])
    LightGBM = Pipeline([
        ('preprocess', preprocessing),
        ('to_df', FunctionTransformer(to_dataframe, validate=False)),
        ('LightGBM', LGBMClassifier(**lgbm_params))
    ])
    LogisticReg = Pipeline([
        ('preprocess', preprocessing),
        ('LogisticRegression', LogisticRegression(**logreg_params))
    ])
    final_est = RandomForestClassifier(**final_params)

    stack = StackingClassifier(
        estimators=[
            ('CatBoost', CatBoost), 
            ('XGBoost', XGBoost), 
            ('LightGBM', LightGBM),
            ('LogisticRegression', LogisticReg)
            ],
        final_estimator=final_est,
        cv=cv,
        n_jobs=-1
    )

    print("\n==== Training the final stacked model ====")
    stack.fit(X_train, y_train)
    evaluate_model(stack, X_test, y_test, "Stacked Model")




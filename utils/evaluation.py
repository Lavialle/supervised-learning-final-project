"""Evaluation utilities for model performance assessment.
Computes:
- Optimal decision threshold based on F1-score
Displays:
- Feature importances
- Classification report
- Confusion matrix
- ROC curve
- Precision-Recall curve
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import pandas as pd
from sklearn.compose import ColumnTransformer

def evaluate_model(model, X_test, y_test, model_name):
    
    # # Get feature importances if available
    if hasattr(model, 'named_steps'):
        preprocessor = model.named_steps['preprocess']  # Allows to access the preprocessing step
        if isinstance(preprocessor, ColumnTransformer):
            features_names = preprocessor.get_feature_names_out()
            print(features_names)
        if model_name in ["RandomForest", "XGBoost", "LightGBM"]:
            feature_importances = model.named_steps[model_name].feature_importances_
        elif model_name == "CatBoost":
            feature_importances = model.named_steps[model_name].get_feature_importance()
        elif model_name == "Logistic":
            feature_importances = model.named_steps[model_name].coef_.flatten()
        else: # If model does not provide feature importances
            feature_importances = None

        # Create a DataFrame to sort importances
        importance_df = pd.DataFrame({
            "Feature": features_names,
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False)

        # Display feature importances as a bar chart
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Feature Importance ({model_name})")
        plt.gca().invert_yaxis()  # Invert the order to display the most important features at the top
        plt.savefig(f"./figures/feature_importances_{model_name}.png", bbox_inches='tight')
        plt.show()

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Decision threshold optimization
    thresholds = np.linspace(0, 1, 100)  # Generate 100 thresholds between 0 and 1
    f1_scores = []
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_test, y_pred_threshold))
    optimal_threshold = thresholds[np.argmax(f1_scores)]  # Threshold that maximizes the F1-score
    print(f"Optimal threshold to maximize F1-score: {optimal_threshold:.2f}")
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label="F1-score", color="blue")
    plt.axvline(optimal_threshold, color="red", linestyle="--", label=f"Optimal Threshold: {optimal_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title(f"F1-score vs. Decision Threshold - {model_name}")
    plt.legend()
    plt.savefig(f"./figures/f1_score_threshold_{model_name}.png", bbox_inches='tight')
    plt.show()
    # Predictions with the optimized threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    # Classification report
    print(f"--- {model_name} ---")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-MDR', 'MDR'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"./figures/confusion_matrix_{model_name}.png", bbox_inches='tight')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Random Chance Line")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.savefig(f"./figures/roc_curve_{model_name}.png", bbox_inches='tight')
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend()
    plt.savefig(f"./figures/precision_recall_curve_{model_name}.png", bbox_inches='tight')
    plt.show()

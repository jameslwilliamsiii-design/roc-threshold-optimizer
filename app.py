# roc_cutoff_optimizer.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    return sensitivity, specificity, accuracy, balanced_acc

def plot_roc(fpr, tpr, auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Load and prepare data
    iris = load_iris()
    X, y = iris.data, iris.target
    y = (y == 2).astype(int)  # Binary: Virginica vs others
    np.random.seed(42)
    X += np.random.normal(0, 0.5, X.shape)  # Add noise

    # Initialize model
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)

    # Cross-validated probability predictions
    y_proba = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]

    # Default threshold (0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)
    sens_d, spec_d, acc_d, bal_acc_d = calculate_metrics(y, y_pred_default)

    # ROC Curve & AUC
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    auc = roc_auc_score(y, y_proba)

    # Optimal threshold: max balanced accuracy
    bal_accuracy = (tpr + (1 - fpr)) / 2
    best_idx = np.argmax(bal_accuracy)
    best_threshold = thresholds[best_idx]
    y_pred_best = (y_proba >= best_threshold).astype(int)
    sens_o, spec_o, acc_o, bal_acc_o = calculate_metrics(y, y_pred_best)

    # Plot ROC
    plot_roc(fpr, tpr, auc)

    # Print metrics
    print("\n--- Metrics at Threshold = 0.5 ---")
    print(f"Sensitivity       : {sens_d:.2f}")
    print(f"Specificity       : {spec_d:.2f}")
    print(f"Accuracy          : {acc_d:.2f}")
    print(f"Balanced Accuracy : {bal_acc_d:.2f}")

    print("\n--- Metrics at Optimal Threshold ---")
    print(f"Optimal Threshold : {best_threshold:.2f}")
    print(f"Sensitivity       : {sens_o:.2f}")
    print(f"Specificity       : {spec_o:.2f}")
    print(f"Accuracy          : {acc_o:.2f}")
    print(f"Balanced Accuracy : {bal_acc_o:.2f}")

if __name__ == "__main__":
    main()

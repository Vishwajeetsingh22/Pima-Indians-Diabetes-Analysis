"""
model_training.py
------------------
Trains a Random Forest classifier on the Pima Indians Diabetes dataset,
evaluates it, saves diagnostic plots (accuracy comparison + confusion
matrix) to the screenshots/ folder, and persists the trained model
(plus the fitted scaler) to outputs/trained_model.pkl.
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from data_preprocessing import preprocess

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """Train several candidate models and return their accuracies + fitted objects."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {"model": model, "accuracy": acc}
        print(f"{name}: accuracy = {acc:.4f}")

    return results


def plot_accuracy_comparison(results, save_path):
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] * 100 for n in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accuracies, color=["#4C72B0", "#DD8452", "#55A868"])
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison — Pima Indians Diabetes Dataset")
    plt.ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            fontweight="bold",
        )
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_test, y_pred, save_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix — Best Model (Random Forest)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_output_summary(y_test, y_pred, y_proba, save_path):
    """A combined 'output' figure: ROC-style probability distribution + class balance."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(
        y_proba[np.array(y_test) == 0], bins=15, alpha=0.7, label="No Diabetes", color="#4C72B0"
    )
    axes[0].hist(
        y_proba[np.array(y_test) == 1], bins=15, alpha=0.7, label="Diabetes", color="#DD8452"
    )
    axes[0].set_xlabel("Predicted probability of Diabetes")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Predicted Probability Distribution")
    axes[0].legend()

    counts = np.bincount(y_pred)
    axes[1].bar(["No Diabetes", "Diabetes"], counts, color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Predicted Class Counts (Test Set)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess()

    results = train_and_compare_models(X_train, X_test, y_train, y_test)

    # Pick the best model by test accuracy
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = results[best_name]["model"]
    best_acc = results[best_name]["accuracy"]
    print(f"\nBest model: {best_name} ({best_acc:.4f} accuracy)")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # Plots
    plot_accuracy_comparison(results, os.path.join(SCREENSHOTS_DIR, "accuracy.png"))
    plot_confusion_matrix(y_test, y_pred, os.path.join(SCREENSHOTS_DIR, "confusion_matrix.png"))
    plot_output_summary(y_test, y_pred, y_proba, os.path.join(SCREENSHOTS_DIR, "output.png"))

    # Persist model + scaler + feature names together
    artifact = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": best_name,
        "test_accuracy": best_acc,
    }
    model_path = os.path.join(OUTPUTS_DIR, "trained_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSaved trained model to {model_path}")

    return results, best_name


if __name__ == "__main__":
    main()

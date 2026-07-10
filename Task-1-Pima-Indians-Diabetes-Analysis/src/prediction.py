"""
prediction.py
-------------
Loads the trained model artifact (outputs/trained_model.pkl) and makes
diabetes-risk predictions for new patient records.

Usage (as a script):
    python prediction.py

Usage (as a module):
    from prediction import predict_diabetes
    result = predict_diabetes({
        "Pregnancies": 2, "Glucose": 130, "BloodPressure": 70,
        "SkinThickness": 22, "Insulin": 85, "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.35, "Age": 29
    })
"""

import os
import pickle
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "trained_model.pkl")


def load_artifact(path: str = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_diabetes(patient: dict, artifact=None):
    """
    Predict diabetes risk for a single patient record.

    Parameters
    ----------
    patient : dict
        Must contain keys: Pregnancies, Glucose, BloodPressure,
        SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    artifact : dict, optional
        Pre-loaded model artifact (model + scaler + feature_names).
        If not given, it is loaded from disk.

    Returns
    -------
    dict with keys: prediction (0/1), label (str), probability (float)
    """
    if artifact is None:
        artifact = load_artifact()

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]

    row = pd.DataFrame([patient])[feature_names]
    row_scaled = scaler.transform(row)

    pred = int(model.predict(row_scaled)[0])
    proba = float(model.predict_proba(row_scaled)[0, 1])

    return {
        "prediction": pred,
        "label": "Diabetic" if pred == 1 else "Not Diabetic",
        "probability": round(proba, 4),
    }


if __name__ == "__main__":
    artifact = load_artifact()
    print(f"Loaded model: {artifact['model_name']} "
          f"(test accuracy = {artifact['test_accuracy']:.4f})\n")

    sample_patients = [
        {
            "Pregnancies": 2, "Glucose": 130, "BloodPressure": 70,
            "SkinThickness": 22, "Insulin": 85, "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.35, "Age": 29,
        },
        {
            "Pregnancies": 8, "Glucose": 183, "BloodPressure": 64,
            "SkinThickness": 29, "Insulin": 155, "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.672, "Age": 45,
        },
    ]

    for i, patient in enumerate(sample_patients, start=1):
        result = predict_diabetes(patient, artifact)
        print(f"Patient {i}: {patient}")
        print(f"  -> Prediction: {result['label']} "
              f"(probability of diabetes = {result['probability']:.2%})\n")

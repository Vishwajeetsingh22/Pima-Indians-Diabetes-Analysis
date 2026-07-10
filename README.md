# Task 1: Pima Indians Diabetes Analysis

An end-to-end machine learning project that predicts whether a patient
has diabetes based on diagnostic health measurements, using the classic
**Pima Indians Diabetes Dataset**.
## Internship Information

This project was completed as part of the **Machine Learning Internship** offered by **CodTech IT Solutions Private Limited**.

- **Intern ID:** CITS2321
- **Task:** Task-1 – Pima Indians Diabetes Analysis
- **Submitted by:** Vishwajeet Singh

## 📁 Project Structure

```
Task-1-Pima-Indians-Diabetes-Analysis/
│
├── data/
│   └── dataset.csv                 # Raw dataset (768 records, 8 features + label)
│
├── notebooks/
│   └── analysis.ipynb              # Full EDA + model training walkthrough (pre-run)
│
├── src/
│   ├── data_preprocessing.py       # Cleaning, imputation, scaling, train/test split
│   ├── model_training.py           # Trains & compares models, saves plots + model
│   └── prediction.py               # Loads saved model and predicts on new patients
│
├── screenshots/
│   ├── accuracy.png                # Model accuracy comparison bar chart
│   ├── confusion_matrix.png        # Confusion matrix of the best model
│   └── output.png                  # Predicted probability / class distribution
│
├── outputs/
│   └── trained_model.pkl           # Pickled best model + scaler + feature names
│
├── documentation/
│   └── project_report.pdf          # Written project report
│
├── requirements.txt
└── README.md
```

## 📊 About the Dataset

The **Pima Indians Diabetes Dataset** (originally from the National
Institute of Diabetes and Digestive and Kidney Diseases) contains
diagnostic measurements for 768 female patients of Pima Indian
heritage, aged 21 and older.

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight in kg/(height in m)^2) |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age in years |
| **Outcome** | **Target: 1 = diabetic, 0 = not diabetic** |

> Note: several columns use `0` to mark a missing measurement
> (physiologically impossible for Glucose, BMI, etc.). These are
> treated as missing and imputed with the column median during
> preprocessing.

## ⚙️ Setup & Installation

1. Clone / download this project folder.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ How to Run

### Option A — Run the scripts directly
```bash
cd src
python data_preprocessing.py   # sanity-check the cleaning/splitting pipeline
python model_training.py       # trains models, saves plots + trained_model.pkl
python prediction.py           # loads the saved model and predicts on sample patients
```

### Option B — Explore the notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```
The notebook is already pre-executed with saved outputs, so you can
also just open and read it without re-running anything.

## 🤖 Models Trained

Three classifiers were trained and compared on an 80/20 stratified
train/test split (features standardized with `StandardScaler`):

| Model | Test Accuracy |
|---|---|
| Logistic Regression | ~70.8% |
| **Support Vector Machine (best)** | **~74.0%** |
| Random Forest | ~73.4% |

The best-performing model (by test accuracy) is automatically selected
and saved to `outputs/trained_model.pkl` along with the fitted
`StandardScaler` and feature list, so `prediction.py` can reproduce the
exact same preprocessing at inference time.

## 📈 Results

- **Accuracy comparison:** `screenshots/accuracy.png`
- **Confusion matrix:** `screenshots/confusion_matrix.png`
- **Prediction output summary:** `screenshots/output.png`

See `documentation/project_report.pdf` for the full write-up
(problem statement, methodology, results, and conclusion).

## 🛠️ Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn (Logistic Regression, SVM, Random Forest)
- matplotlib, seaborn (visualization)
- Jupyter Notebook

## 📌 Notes for Submission

- This folder is self-contained and can be zipped/pushed to GitHub as-is.
- `outputs/trained_model.pkl` is already generated — no need to retrain
  before submitting, but re-running `model_training.py` will regenerate
  it deterministically (fixed `random_state=42`).

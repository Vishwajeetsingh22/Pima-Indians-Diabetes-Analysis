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

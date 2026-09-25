import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data_path = "data/student_exam_scores.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# ==========================================
# 3. MISSING VALUES
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================
# 4. FEATURE SELECTION
# ==========================================

features = [
    "Study_Hours",
    "Attendance_Percentage",
    "Previous_Exam_Marks",
    "Assignment_Marks",
    "Internal_Marks",
    "Practice_Test_Score",
    "Sleep_Hours"
]

target = "Final_Exam_Marks"


X = df[features]
y = df[target]


# ==========================================
# 5. NUMERICAL FEATURES
# ==========================================

numeric_features = features


# ==========================================
# 6. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    )
])


# ==========================================
# 7. LINEAR REGRESSION MODEL
# ==========================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "regressor",
        LinearRegression()
    )
])


# ==========================================
# 8. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 9. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ==========================================
# 10. PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 11. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========================================")
print("          MODEL EVALUATION")
print("========================================")

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

print("========================================")


# ==========================================
# 12. SAVE MODEL
# ==========================================

model_path = "model/student_marks_model.pkl"

joblib.dump(
    model,
    model_path
)

print("\nModel saved successfully!")
print("Model location:", model_path)
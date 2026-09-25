# Student Marks Prediction using Machine Learning

## Project Objective

The objective of this project is to build a Machine Learning model
that predicts a student's final exam marks based on academic and
study-related factors.

## Features

- Study Hours
- Attendance Percentage
- Previous Exam Marks
- Sleep Hours

## Target Variable

- Final Exam Marks

## Machine Learning Algorithm

Linear Regression

## Dataset

The dataset was obtained from Kaggle.

Dataset:
Student Exam Scores & Study Habits Dataset

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset
2. Checked missing values
3. Removed duplicate records
4. Selected relevant features
5. Handled missing numerical values using median imputation
6. Encoded categorical variables if present

## Model Training

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

A Linear Regression model was trained on the training dataset.

## Model Evaluation

The model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Run `train_model.py` to generate the actual evaluation values.

## Model Saving

The trained model is saved using Joblib:

model/student_marks_model.pkl

## Streamlit Application

The Streamlit application allows users to enter:

- Study Hours
- Attendance
- Previous Exam Marks
- Sleep Hours

The application then displays:

- Predicted Final Exam Marks
- Input Summary
- Basic Interpretation

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
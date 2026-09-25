import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓",
    layout="centered"
)

model = joblib.load("model/student_marks_model.pkl")

st.title("🎓 Student Marks Prediction")
st.write(
    "Enter the student's academic and study-related details "
    "to predict the final exam marks."
)

st.header("📋 Student Details")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

previous_scores = st.number_input(
    "Previous Exam Marks",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

assignment_marks = st.number_input(
    "Assignment Marks",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

internal_marks = st.number_input(
    "Internal Marks",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

practice_test = st.number_input(
    "Practice Test Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0,
    step=0.5
)

if st.button("🔮 Predict Final Marks"):

    input_data = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance_Percentage": [attendance],
        "Previous_Exam_Marks": [previous_scores],
        "Assignment_Marks": [assignment_marks],
        "Internal_Marks": [internal_marks],
        "Practice_Test_Score": [practice_test],
        "Sleep_Hours": [sleep_hours]
    })

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.success(
        f"🎯 Predicted Final Exam Marks: {prediction:.2f} / 100"
    )

    st.subheader("📊 Input Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Study Hours",
            "Attendance",
            "Previous Exam Marks",
            "Assignment Marks",
            "Internal Marks",
            "Practice Test Score",
            "Sleep Hours"
        ],
        "Value": [
            study_hours,
            f"{attendance}%",
            previous_scores,
            assignment_marks,
            internal_marks,
            practice_test,
            sleep_hours
        ]
    })

    st.table(summary)

    st.subheader("💡 Interpretation")

    if prediction >= 75:
        st.info(
            "The predicted marks are relatively high "
            "based on the entered student details."
        )
    elif prediction >= 50:
        st.info(
            "The predicted marks are in the moderate range."
        )
    else:
        st.warning(
            "The predicted marks are relatively low. "
            "Improving study habits and academic performance may help."
        )
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import Axes3D

# Set page configuration
st.set_page_config(page_title="Student Performance Prediction", layout="centered")

# Load the dataset
DATA_PATH = "assgmt01_student_performance_dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Handle missing values
    df["study_hours_per_week"].fillna(df["study_hours_per_week"].mean(), inplace=True)
    df["attendance_rate"].fillna(df["attendance_rate"].mean(), inplace=True)
    df["previous_exam_scores"].fillna(df["previous_exam_scores"].median(), inplace=True)
    df["assignments_completed"].fillna(df["assignments_completed"].median(), inplace=True)
    df["extracurricular_participation"].fillna(df["extracurricular_participation"].mode()[0], inplace=True)

    # Feature Engineering
    df["study_hours_per_week_squared"] = df["study_hours_per_week"] ** 2
    df["attendance_rate_squared"] = df["attendance_rate"] ** 2
    df["study_attendance_interaction"] = df["study_hours_per_week"] * df["attendance_rate"]
    df["assignments_per_week"] = df["assignments_completed"] / 7

    return df

# Load and preprocess the dataset
df = load_data()

# Define features and target variable
features = [
    "study_hours_per_week", "attendance_rate", "previous_exam_scores",
    "assignments_completed", "extracurricular_participation",
    "study_hours_per_week_squared", "attendance_rate_squared",
    "study_attendance_interaction", "assignments_per_week"
]
X = df[features]
y = df["final_exam_score"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

# Initialize and train Lasso Regression model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_imputed, y_train)

# Function to reset form fields
def reset_fields():
    st.session_state.clear()
    st.session_state["study_hours"] = 10.0
    st.session_state["attendance_rate"] = 75
    st.session_state["previous_exam_scores"] = 60.0
    st.session_state["assignments_completed"] = 80
    st.session_state["extracurricular"] = "No"

# Initialize session state
if "study_hours" not in st.session_state:
    reset_fields()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Graph", "Data"])

if page == "Home":
    st.title("🎓 Student Performance Prediction")

    # Input fields on main screen
    study_hours = st.number_input("Study Hours per Week:", min_value=0.0, max_value=100.0, value=st.session_state["study_hours"], key="study_hours")
    attendance_rate = st.slider("Attendance Rate (%):", min_value=0, max_value=100, value=st.session_state["attendance_rate"], key="attendance_rate")
    previous_exam_scores = st.number_input("Previous Exam Scores:", min_value=0.0, max_value=100.0, value=st.session_state["previous_exam_scores"], key="previous_exam_scores")
    assignments_completed = st.number_input("Assignments Completed:", min_value=0, max_value=100, value=st.session_state["assignments_completed"], key="assignments_completed")
    extracurricular = st.radio("Extracurricular Participation:", options=["No", "Yes"], index=0 if st.session_state["extracurricular"] == "No" else 1, key="extracurricular")

    extracurricular = 1 if extracurricular == "Yes" else 0

    # Centered buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        clear_button = st.button("Clear", use_container_width=True)
    with col2:
        predict_button = st.button("Predict", use_container_width=True)

    # Clear button functionality
    if clear_button:
        reset_fields()
        st.experimental_rerun() 

    # Predict button functionality
    if predict_button:
        try:
            # Feature Engineering for Input Data
            study_hours_squared = study_hours ** 2
            attendance_rate_squared = attendance_rate ** 2
            study_attendance_interaction = study_hours * attendance_rate
            assignments_per_week = assignments_completed / 7

            input_data = pd.DataFrame({
                "study_hours_per_week": [study_hours],
                "attendance_rate": [attendance_rate],
                "previous_exam_scores": [previous_exam_scores],
                "assignments_completed": [assignments_completed],
                "extracurricular_participation": [extracurricular],
                "study_hours_per_week_squared": [study_hours_squared],
                "attendance_rate_squared": [attendance_rate_squared],
                "study_attendance_interaction": [study_attendance_interaction],
                "assignments_per_week": [assignments_per_week]
            })

            input_data_scaled = scaler.transform(input_data)
            input_data_imputed = imputer.transform(input_data_scaled)
            lasso_pred = lasso_model.predict(input_data_imputed)[0]
            st.info(f"Prediction Result: {lasso_pred:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif page == "Graph":
    st.title("📊 Data Visualizations")

    # Histogram
    st.subheader("Distribution of Final Exam Scores")
    plt.figure(figsize=(8, 5))
    sns.histplot(df['final_exam_score'], kde=True)
    st.pyplot(plt)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(plt)

    # 3D Scatter Plot
    st.subheader("3D Scatter Plot")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['study_hours_per_week'], df['attendance_rate'], df['final_exam_score'], c='b', marker='o')
    ax.set_xlabel('Study Hours per Week')
    ax.set_ylabel('Attendance Rate (%)')
    ax.set_zlabel('Final Exam Score')
    st.pyplot(fig)

    # Box Plot
    st.subheader("Box Plot of Final Exam Scores by Extracurricular Participation")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='extracurricular_participation', y='final_exam_score', data=df)
    st.pyplot(plt)

elif page == "Data":
    st.title("📄 Dataset")
    st.dataframe(df)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Designed by Shahidatul Hidayah © 2024")

# Dark Theme Styling (CSS)
st.markdown(
    """
    <style>
    body {
        color: #f0f0f0;
        background-color: #1e1e1e;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #800d2f;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

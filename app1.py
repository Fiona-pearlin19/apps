import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load and Train Model
# -----------------------------
data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter the patient details below and click **Predict**."
)

st.write(f"### Model Accuracy: **{accuracy:.2%}**")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
mean_radius = st.number_input("Mean Radius", 0.0, 50.0, 14.0)
mean_texture = st.number_input("Mean Texture", 0.0, 50.0, 20.0)
mean_perimeter = st.number_input("Mean Perimeter", 0.0, 200.0, 90.0)
mean_area = st.number_input("Mean Area", 0.0, 3000.0, 650.0)
mean_smoothness = st.number_input("Mean Smoothness", 0.0, 1.0, 0.1)
mean_compactness = st.number_input("Mean Compactness", 0.0, 1.0, 0.1)
mean_concavity = st.number_input("Mean Concavity", 0.0, 1.0, 0.1)
mean_concave_points = st.number_input("Mean Concave Points", 0.0, 1.0, 0.05)
mean_symmetry = st.number_input("Mean Symmetry", 0.0, 1.0, 0.2)
mean_fractal_dimension = st.number_input("Mean Fractal Dimension", 0.0, 1.0, 0.06)
radius_error = st.number_input("Radius Error", 0.0, 5.0, 0.5)
texture_error = st.number_input("Texture Error", 0.0, 5.0, 1.0)
perimeter_error = st.number_input("Perimeter Error", 0.0, 30.0, 3.0)
area_error = st.number_input("Area Error", 0.0, 600.0, 40.0)
smoothness_error = st.number_input("Smoothness Error", 0.0, 1.0, 0.01)
compactness_error = st.number_input("Compactness Error", 0.0, 1.0, 0.02)
concavity_error = st.number_input("Concavity Error", 0.0, 1.0, 0.02)
concave_points_error = st.number_input("Concave Points Error", 0.0, 1.0, 0.01)
symmetry_error = st.number_input("Symmetry Error", 0.0, 1.0, 0.02)
fractal_dimension_error = st.number_input("Fractal Dimension Error", 0.0, 1.0, 0.003)
worst_radius = st.number_input("Worst Radius", 0.0, 50.0, 16.0)
worst_texture = st.number_input("Worst Texture", 0.0, 60.0, 25.0)
worst_perimeter = st.number_input("Worst Perimeter", 0.0, 300.0, 110.0)
worst_area = st.number_input("Worst Area", 0.0, 5000.0, 900.0)
worst_smoothness = st.number_input("Worst Smoothness", 0.0, 1.0, 0.15)
worst_compactness = st.number_input("Worst Compactness", 0.0, 2.0, 0.2)
worst_concavity = st.number_input("Worst Concavity", 0.0, 2.0, 0.2)
worst_concave_points = st.number_input("Worst Concave Points", 0.0, 1.0, 0.1)
worst_symmetry = st.number_input("Worst Symmetry", 0.0, 1.0, 0.3)
worst_fractal_dimension = st.number_input("Worst Fractal Dimension", 0.0, 1.0, 0.08)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    sample = [[
        mean_radius,
        mean_texture,
        mean_perimeter,
        mean_area,
        mean_smoothness,
        mean_compactness,
        mean_concavity,
        mean_concave_points,
        mean_symmetry,
        mean_fractal_dimension,
        radius_error,
        texture_error,
        perimeter_error,
        area_error,
        smoothness_error,
        compactness_error,
        concavity_error,
        concave_points_error,
        symmetry_error,
        fractal_dimension_error,
        worst_radius,
        worst_texture,
        worst_perimeter,
        worst_area,
        worst_smoothness,
        worst_compactness,
        worst_concavity,
        worst_concave_points,
        worst_symmetry,
        worst_fractal_dimension
    ]]

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)

    if prediction[0] == 1:
        st.success("Prediction: ❤️ No Heart Disease")
    else:
        st.error("Prediction: ⚠️ Heart Disease Detected")

    st.subheader("Prediction Confidence")

    st.write({
        "Heart Disease": f"{probability[0][0]*100:.2f}%",
        "No Heart Disease": f"{probability[0][1]*100:.2f}%"
    })

    st.progress(float(max(probability[0])))

st.divider()

st.caption("Developed using Streamlit and Scikit-Learn")

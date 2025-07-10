import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load the trained pipeline
@st.cache_resource
def load_model():
    return joblib.load("D:\Development\Programming\AI\Sprints Final Project\heart_disease_models\RandomForest_pipeline.pkl")

model = load_model()

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.title("💓 Heart Disease Risk Predictor")
st.markdown("This app uses a trained machine learning model to assess your risk of heart disease.")

# Sidebar inputs
st.sidebar.header("🔎 Enter Patient Data")

# Input form
def get_user_input():
    age = st.sidebar.slider("Age", 20, 100, 55)
    sex = st.sidebar.radio("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
    chol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 600, 250)
    fbs = st.sidebar.checkbox("Fasting Blood Sugar > 120 mg/dL")
    restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 160)
    exang = st.sidebar.checkbox("Exercise Induced Angina")
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0, step=0.1)
    ca = st.sidebar.slider("Number of Major Vessels (0–3)", 0, 3, 0)
    thal = st.sidebar.selectbox("Thalassemia", [1, 2, 3])  # Assuming numerical encoding

    # Format input as DataFrame
    input_data = pd.DataFrame([{
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs else 0,
        'restecg': restecg,
        'thalach': thalach,
        'exang': 1 if exang else 0,
        'oldpeak': oldpeak,
        'ca': ca,
        'thal': thal
    }])

    return input_data

input_df = get_user_input()

# Predict and display results
if st.button("💡 Predict Risk"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("🩺 Prediction Result")
        st.markdown(f"### {'⚠️ High Risk' if prediction == 1 else '✅ Low Risk'}")
        st.markdown(f"**Probability of Heart Disease:** `{probability * 100:.2f}%`")

        # Pie Chart
        fig = px.pie(
            names=["No Disease", "Disease"],
            values=[1 - probability, probability],
            color_discrete_sequence=["#2ECC71", "#E74C3C"]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Health Advice
        with st.expander("💬 Health Recommendations"):
            if prediction == 1:
                st.warning("Consult a cardiologist. Consider improving diet, exercise, and stress levels.")
            else:
                st.success("Keep up your good health with regular checkups and a healthy lifestyle.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

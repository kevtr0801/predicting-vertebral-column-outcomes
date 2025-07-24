import streamlit as st
import numpy as np
import joblib

# Load the saved pipeline and label encoder
pipeline = joblib.load('vertebral_voting_hard_svm-rf-logreg_smote_v1.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="Vertebral Classifier", page_icon="🦴", layout="centered")

st.title("🦴 Vertebral Column Condition Classifier")
st.markdown("""
This app predicts whether a patient has:
- **Hernia**
- **Normal spine**
- **Spondylolisthesis**

based on spinal geometry measurements.
""")

# Display the info with columns 
with st.form("prediction_form"):
    st.subheader("🔢 Enter Patient Measurements")

    col1, col2 = st.columns(2)

    with col1:
        pelvic_incidence = st.number_input(
            "Pelvic Incidence",
            min_value=0.0,
            max_value=120.0,
            value=60.0,
            step=0.1,
            help="Angle between sacral plate and femur head axis"
        )
        pelvic_tilt = st.number_input(
            "Pelvic Tilt",
            min_value=-20.0,
            max_value=40.0,
            value=17.0,
            step=0.1,
            help="Angle of pelvis rotation"
        )
        lumbar_lordosis_angle = st.number_input(
            "Lumbar Lordosis Angle",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.1,
            help="Curvature of the lower spine"
        )

    with col2:
        sacral_slope = st.number_input(
            "Sacral Slope",
            min_value=0.0,
            max_value=90.0,
            value=43.0,
            step=0.1,
            help="Angle of the top of the sacrum"
        )
        pelvic_radius = st.number_input(
            "Pelvic Radius",
            min_value=50.0,
            max_value=150.0,
            value=120.0,
            step=0.1,
            help="Distance from sacral endplate to femur axis"
        )
        degree_spondylolisthesis = st.number_input(
            "Degree of Spondylolisthesis",
            min_value=-50.0,
            max_value=200.0,
            value=20.0,
            step=0.1,
            help="Forward slip percentage of vertebra"
        )

    submit = st.form_submit_button("🔍 Predict")

if submit:
    user_input = np.array([[pelvic_incidence, pelvic_tilt, lumbar_lordosis_angle,
                            sacral_slope, pelvic_radius, degree_spondylolisthesis]])

    prediction = pipeline.predict(user_input)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    st.success(f"🩺 **Predicted Condition:** `{predicted_label}`")

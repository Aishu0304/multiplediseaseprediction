import os
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Get the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved models
diabetes_model = pickle.load(open('C:/Users/aishu/OneDrive/Desktop/Multiplediseaseprediction/savedmodels/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('C:/Users/aishu/OneDrive/Desktop/Multiplediseaseprediction/savedmodels/heart_disease_model.sav', 'rb'))

# Load the mental health model
mental_health_model = pickle.load(open('C:/Users/aishu/OneDrive/Desktop/Multiplediseaseprediction/savedmodels/mental.sav', 'rb'))

# Load the encoders
with open('C:/Users/aishu/OneDrive/Desktop/Multiplediseaseprediction/savedmodels/labels.pkl', 'rb') as f:
    categorical_encoders = pickle.load(f)



# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Mental Health Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'brain'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=110)
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0, max_value=200, value=80)
    with col1:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0, max_value=100, value=20)
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=30)
    with col3:
        BMI = st.number_input('BMI value', min_value=0.0, max_value=100.0, value=25.0)
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=2.5, value=0.5)
    with col2:
        Age = st.number_input('Age of the Person', min_value=0, max_value=120, value=25)

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        diab_prediction = diabetes_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'
    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=25)
    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'])
    with col3:
        cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=200, value=120)
    with col2:
        chol = st.number_input('Serum Cholesterol in mg/dl', min_value=0, max_value=600, value=200)
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
    with col1:
        restecg = st.selectbox('Resting Electrocardiographic results', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=0, max_value=250, value=150)
    with col3:
        exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=10.0, value=1.0)
    with col2:
        slope = st.selectbox('Slope of the peak exercise ST segment', ['Upsloping', 'Flat', 'Downsloping'])
    with col3:
        ca = st.number_input('Major vessels colored by fluoroscopy', min_value=0, max_value=4, value=0)
    with col1:
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        heart_prediction = heart_disease_model.predict([user_input])
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'
    st.success(heart_diagnosis)

# Mental Health Prediction Page
# Function to encode user input for the mental health model
def encode_input(user_input):
    encoded_input = [
        user_input['Age'],  # Age is a numerical feature and doesn't need encoding
        categorical_encoders['Gender'].transform([0 if user_input['Gender'] == 'Male' else 1])[0],
        categorical_encoders['family_history'].transform([1 if user_input['family_history'] == 'Yes' else 0])[0],
        categorical_encoders['benefits'].transform([1 if user_input['benefits'] == 'Yes' else 0])[0],
        categorical_encoders['care_options'].transform([1 if user_input['care_options'] == 'Yes' else 0])[0]
    ]
    return np.array([encoded_input])

# Mental Health Prediction Page
if selected == 'Mental Health Prediction':
    st.title("Mental Health Prediction using ML")

    # Input fields
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    family_history = st.selectbox('Family History', ['Yes', 'No'])
    benefits = st.selectbox('Benefits', ['Yes', 'No'])
    care_options = st.selectbox('Care Options', ['Yes', 'No'])

    # Predict button
    if st.button('Predict'):
        user_input = {
            'Age': age,
            'Gender': gender,
            'family_history': family_history,
            'benefits': benefits,
            'care_options': care_options
        }
        encoded_input = encode_input(user_input)
        if encoded_input is not None:
            prediction = mental_health_model.predict(encoded_input)
            result = "Needs mental health treatment" if prediction[0] == 1 else "Does not need mental health treatment"
            st.success(result)


import streamlit as st
import joblib
import pandas as pd
import numpy as np

def load_model():
    model = joblib.load('model_outputs\best_model.joblib')
    scaler = joblib.load('model_outputs\scaler.joblib')
    selected_features = joblib.load('model_outputs\selected_features.joblib')
    return model, scaler, selected_features
    
model, scaler, selected_features = load_model()

if model is None:
    st.stop()
  
st.title('Fetal Health Prediction')
st.write('Enter cardiotocogram (CTG) measurements to predict fetal health status')

st.sidebar.header('About')
st.sidebar.write('This app predicts fetal health using machine learning')
st.sidebar.write('**Model:** Gradient Boosting Classifier')
st.sidebar.write('**Accuracy:** 94.56%')

st.header('📊 Enter CTG Measurements')

col1, col2 = st.columns(2)

with col1:
    baseline_value = st.slider('Baseline Heart Rate (bpm)', 100, 160, 133, step=1)
    accelerations = st.slider('Accelerations', 0.0, 0.02, 0.005, step=0.001)
    fetal_movement = st.slider('Fetal Movement', 0.0, 0.4, 0.1, step=0.01)
    severe_decelerations = st.slider('Severe Decelerations', 0.0, 0.5, 0.0, step=0.01)
    prolongued_decelerations = st.slider('Prolonged Decelerations', 0.0, 0.5, 0.0, step=0.01)

with col2:
    abnormal_short_term_variability = st.slider('Abnormal Short Term Variability', 0.0, 100.0, 30.0, step=1.0)
    mean_value_of_short_term_variability = st.slider('Mean Short Term Variability', 0.0, 200.0, 50.0, step=1.0)
    percentage_of_time_with_abnormal_long_term_variability = st.slider('% Abnormal Long Term Variability', 0.0, 100.0, 30.0, step=1.0)
    mean_value_of_long_term_variability = st.slider('Mean Long Term Variability', 0.0, 200.0, 50.0, step=1.0)
    histogram_width = st.slider('Histogram Width', 0.0, 400.0, 100.0, step=10.0)

if st.button('🔍 Make Prediction', key='predict'):
    # Prepare input
    input_dict = {
        'baseline value': baseline_value,
        'accelerations': accelerations,
        'fetal_movement': fetal_movement,
        'severe_decelerations': severe_decelerations,
        'prolongued_decelerations': prolongued_decelerations,
        'abnormal_short_term_variability': abnormal_short_term_variability,
        'mean_value_of_short_term_variability': mean_value_of_short_term_variability,
        'percentage_of_time_with_abnormal_long_term_variability': percentage_of_time_with_abnormal_long_term_variability,
        'mean_value_of_long_term_variability': mean_value_of_long_term_variability,
        'histogram_width': histogram_width,
    }
    
    input_data = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.header('📋 Prediction Results')
    
    class_names = {1: '🟢 Normal', 2: '🟡 Suspect', 3: '🔴 Pathological'}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric('Prediction', class_names[prediction])
    
    with col2:
        st.metric('Confidence', f'{probability[prediction-1]:.1%}')
    
    with col3:
        st.metric('Status', 'High Risk' if prediction > 1 else 'Low Risk')

    st.info(f"**Status:** {class_descriptions[prediction]}")
    
    st.subheader('Class Probabilities')
    prob_data = pd.DataFrame({
        'Class': ['Normal', 'Suspect', 'Pathological'],
        'Probability': probability
    })
    st.bar_chart(prob_data.set_index('Class'))
    
    st.warning('⚠️ Disclaimer: This is a decision support tool. Always consult qualified healthcare professionals.')

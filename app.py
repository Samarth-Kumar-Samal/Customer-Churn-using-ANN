import streamlit as st
import pickle
import numpy as np
import tensorflow as tf

# Load the model and scaler
model = tf.keras.models.load_model('ann_model.h5')
scaler = pickle.load(open('Scaler.pkl', 'rb'))

def predict(data):
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    pred = model.predict(final_input)[0]
    output = np.where(pred < 0.5, 0, 1)
    return output[0]

def main():

    # Setting up the page configutation
    st.set_page_config(
    page_title='Customer Churn Prediction Application',
    page_icon='💻',
    layout='wide'
    )

    st.title('💻Customer Churn Prediction')

    # Form to take input
    st.header('Predict using Form')
    form_data = {}
    headings = ["Gender", "SeniorCitizen", "Partner", "Dependents", "Tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 
                "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
    
    for feature in headings:
        # Add a unique key for each st.number_input
        form_data[feature] = st.number_input(f'Enter {feature}', value=0.0, key=f'{feature}_form')

    if st.button('Submit'):
        prediction = predict(list(form_data.values()))
        st.success(f'The Customer Churn prediction is {prediction}')

if __name__ == "__main__":
    main()

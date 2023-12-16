import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open('ANN_Model.pkl', 'rb'))
scaler = pickle.load(open('Scaler.pkl', 'rb'))

def predict_api(data):
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    pred = model.predict(new_data)
    output = np.where(pred < 0.5,0,1)
    return output[0][0]

def predict(data):
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    pred = model.predict(final_input)[0]
    output = np.where(pred < 0.5,0,1)
    return output[0]

def main():
    st.title('Customer Churn Prediction')

    # Sidebar
    st.sidebar.header('Input Data')
    input_data = {}
    headings = ["Gender", "SeniorCitizen", "Partner", "Dependents", "Tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 
                "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
    for feature in headings:
        # Add a unique key for each st.number_input
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}', value=0.0, key=f'{feature}_sidebar')

    # Predict using API
    if st.sidebar.button('Predict'):
        prediction = predict_api(input_data)
        st.sidebar.success(f'The Customer Churn prediction is {prediction}')

    # Predict using form
    st.header('Predict using Form')
    form_data = {}
    for feature in headings:
        # Add a unique key for each st.number_input
        form_data[feature] = st.number_input(f'Enter {feature}', value=0.0, key=f'{feature}_form')

    if st.button('Submit'):
        prediction = predict(list(form_data.values()))
        st.success(f'The Customer Churn prediction is {prediction}')

if __name__ == "__main__":
    main()

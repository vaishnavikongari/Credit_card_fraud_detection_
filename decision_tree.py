import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score

# Load the trained Decision Tree model
model = pickle.load(open('decision_tree_model.pkl', 'rb'))

# UI Layout
st.title("Credit Card Fraud Detection Using Decision Tree")
st.write("""Upload a CSV file containing transaction data. The system will analyze and detect fraudulent transactions.""")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    
    # Preprocessing
    st.write("Uploaded Data:")
    st.dataframe(data.head())
    
    # Check if 'Amount' column is present
    if 'Amount' in data.columns:
        # Standardize the 'Amount' column
        scaler = StandardScaler()
        data['Amount'] = scaler.fit_transform(data[['Amount']])
    else:
        st.error("'Amount' column is missing in the uploaded file!")
    
    # Ensure the target column 'Class' exists
    if 'Class' in data.columns:
        # Define X and y
        X = data.drop(['Class'], axis=1)  # Features (all columns except 'Class')
        y = data['Class']  # Target (fraudulent or not)
        
        # Prediction using the trained model
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        # Calculate the accuracy of the model
        # accuracy = accuracy_score(y, predictions)
        # st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
        auprc = average_precision_score(y, probabilities)
        st.write(f"Model AUPRC Score: {auprc:.4f}")
        
        # Display Results
        data['Fraud_Probability'] = probabilities
        data['Prediction'] = predictions
        st.write("Analyzed Data with Predictions:")
        st.dataframe(data[['Fraud_Probability', 'Prediction']])
        
        # Fraud Detection Summary
        st.write("Fraud Detection Summary:")
        total_transactions = len(data)
        fraudulent_transactions = sum(predictions)
        st.write(f"Total Transactions: {total_transactions}")
        st.write(f"Fraudulent Transactions Detected: {fraudulent_transactions}")
        st.write(f"Fraud Percentage: {fraudulent_transactions / total_transactions * 100:.2f}%")
        
    else:
        st.error("'Class' column is missing in the uploaded file!")
    
    data_fraud = data[data['Class'] == 1]
    data_non_fraud = data[data['Class'] == 0]
    
    if st.button('Visualize'):
        # Plotting the distribution of Fraud vs Non-Fraud
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data_fraud['Time'], label='Fraudulent', shade=True)
        sns.kdeplot(data_non_fraud['Time'], label='Non-Fraudulent', shade=True)
        plt.xlabel('Seconds elapsed between the transaction and the first transaction')
        plt.title('Distribution of Transaction Times')
        st.pyplot(plt)

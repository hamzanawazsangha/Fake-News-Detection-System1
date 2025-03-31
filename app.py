import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data
nltk.download('punkt')

# Set page config
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create suggestions directory if it doesn't exist
if not os.path.exists("suggestions"):
    os.makedirs("suggestions")

# Function to save comments
def save_comment(name, email, comment):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("suggestions/comments.txt", "a") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Name: {name}\n")
        f.write(f"Email: {email}\n")
        f.write(f"Comment: {comment}\n")
        f.write("-" * 50 + "\n\n")

# Load models
@st.cache_resource
def load_models():
    models = {
        'Logistic Regression': pickle.load(open('assets/models/Logistic Regression.pkl', 'rb')),
        'Random Forest': pickle.load(open('assets/models/Random Forest.pkl', 'rb')),
        'Support Vector Machine': pickle.load(open('assets/models/SVC.pkl', 'rb')),
        'K-Nearest Neighbors': pickle.load(open('assets/models/KNeighbors Classifier.pkl', 'rb')),
        'XGBoost': pickle.load(open('assets/models/XGBOOST.pkl', 'rb')),
        'Neural Network (Best Model)': load_model('assets/models/Fakenews.h5')
    }
    return models

models = load_models()

# Load evaluation data
@st.cache_data
def load_evaluation_data():
    evaluation_data = {
        'accuracy': {
            'Logistic Regression': 0.99,
            'Random Forest': 0.99,
            'Support Vector Machine': 0.99,
            'K-Nearest Neighbors': 0.98,
            'XGBoost': 0.99,
            'Neural Network (Best Model)': 0.99
        },
        'classification_reports': {
            'Logistic Regression': pickle.load(open('assets/classification report/classification_LR.pkl', 'rb')),
            'Random Forest': pickle.load(open('assets/classification report/classification_RF.pkl', 'rb')),
            'Support Vector Machine': pickle.load(open('assets/classification report/classification_svc.pkl', 'rb')),
            'K-Nearest Neighbors': pickle.load(open('assets/classification report/classification_KNN.pkl', 'rb')),
            'XGBoost': pickle.load(open('assets/classification report/classification_xgb.pkl', 'rb')),
            'Neural Network (Best Model)': pickle.load(open('assets/classification report/classification_Neural.pkl', 'rb'))
        },
        'roc_data': {
            'Logistic Regression': pickle.load(open('assets/roc_curve/roc_data_LR.pkl', 'rb')),
            'Random Forest': pickle.load(open('assets/roc_curve/roc_data_RF.pkl', 'rb')),
            'Support Vector Machine': pickle.load(open('assets/roc_curve/roc_data_svc.pkl', 'rb')),
            'K-Nearest Neighbors': pickle.load(open('assets/roc_curve/roc_data_KNN.pkl', 'rb')),
            'XGBoost': pickle.load(open('assets/roc_curve/roc_data_xgb.pkl', 'rb')),
            'Neural Network (Best Model)': pickle.load(open('assets/roc_curve/roc_data_neural.pkl', 'rb'))
        },
        'confusion_matrices': {
            'Logistic Regression': pickle.load(open('assets/confusion matrix/confusion_LR.pkl', 'rb')),
            'Random Forest': pickle.load(open('assets/confusion matrix/confusion_RF.pkl', 'rb')),
            'Support Vector Machine': pickle.load(open('assets/confusion matrix/confusion_svc.pkl', 'rb')),
            'K-Nearest Neighbors': pickle.load(open('assets/confusion matrix/confusion_KNN.pkl', 'rb')),
            'XGBoost': pickle.load(open('assets/confusion matrix/confusion_xgb.pkl', 'rb')),
            'Neural Network (Best Model)': pickle.load(open('assets/confusion matrix/confusion_Neural.pkl', 'rb'))
        }
    }
    return evaluation_data

evaluation_data = load_evaluation_data()

# Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    return ' '.join(tokens)

# Load Vectorizer
@st.cache_resource
def load_vectorizer():
    vectorizer = TfidfVectorizer(max_features=100)
    # Initialize with some dummy data
    dummy_data = ["fake news", "real news", "political news", "world news"]
    vectorizer.fit(dummy_data)
    return vectorizer

vectorizer = load_vectorizer()

# Prediction function
def predict_news(text, model_name):
    try:
        processed_text = preprocess_text(text)
        embedding = vectorizer.transform([processed_text]).toarray()
        
        if model_name == 'Neural Network (Best Model)':
            prediction = models[model_name].predict(embedding)
            prediction = (prediction > 0.5).astype(int)[0][0]
        else:
            prediction = models[model_name].predict(embedding)[0]
        
        return "Fake" if prediction == 0 else "Real"
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Main app function
def main():
    st.title("📰 Fake News Detection System")
    
    # Introduction section
    st.header("Introduction")
    st.markdown("""
    Welcome to the Fake News Detection System! This application uses machine learning and deep learning 
    models to classify news articles as either **Real** or **Fake**.
    """)
    
    # Instructions section
    st.header("Instructions")
    st.markdown("""
    1. Enter or paste a news article text in the input box below
    2. Select which model you'd like to use for prediction
    3. Click the "Predict" button
    4. View the prediction result
    """)
    
    # Prediction section
    st.header("News Classification")
    news_text = st.text_area("Enter the news article text:", height=200)
    
    model_options = [
        'Logistic Regression (Accuracy: 99%)',
        'Random Forest (Accuracy: 99%)',
        'Support Vector Machine (Accuracy: 99%)',
        'K-Nearest Neighbors (Accuracy: 98%)',
        'XGBoost (Accuracy: 99%)',
        'Neural Network (Best Model) (Accuracy: 99%)'
    ]
    selected_model = st.selectbox("Choose a model:", model_options)
    model_name = selected_model.split(' (Accuracy:')[0]
    
    if st.button("Predict"):
        if news_text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                prediction = predict_news(news_text, model_name)
                if prediction == "Fake":
                    st.error(f"🔴 This news is classified as **{prediction}**")
                else:
                    st.success(f"🟢 This news is classified as **{prediction}**")
    
    # Developer Hub section
    st.markdown("---")
    if st.button("Developer Hub"):
        st.header("🧑‍💻 Developer Hub")
        
        # Model accuracy comparison
        st.subheader("Model Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(evaluation_data['accuracy'].keys()), y=list(evaluation_data['accuracy'].values()))
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Classification report
        st.subheader(f"Classification Report: {model_name}")
        report_df = evaluation_data['classification_reports'][model_name]
        st.dataframe(report_df)
        
        # ROC curve
        st.subheader(f"ROC Curve: {model_name}")
        roc_data = evaluation_data['roc_data'][model_name]
        fig, ax = plt.subplots()
        ax.plot(roc_data['fpr'], roc_data['tpr'], label=f"AUC = {roc_data['roc_auc']:.2f}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        st.pyplot(fig)
        
        # Confusion matrix
        st.subheader(f"Confusion Matrix: {model_name}")
        cm = evaluation_data['confusion_matrices'][model_name]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    # Comment/Suggestion Section
    st.markdown("---")
    st.header("💬 Suggestions & Feedback")
    with st.form("comment_form"):
        name = st.text_input("Your Name (optional)")
        email = st.text_input("Your Email (optional)")
        comment = st.text_area("Your Feedback", height=150)
        if st.form_submit_button("Submit"):
            if comment.strip():
                save_comment(name, email, comment)
                st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()

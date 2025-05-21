import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Set page config with custom CSS for a beautiful gradient background
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(120deg, #e0eafc, #cfdef3);
        background-attachment: fixed;
    }

    .title {
        color: #2c3e50;
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 30px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.08);
        padding: 20px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }

    .stTextArea textarea {
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #ced4da;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        font-size: 16px;
    }

    .stButton>button {
        background-color: #1abc9c;
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 12px rgba(26, 188, 156, 0.3);
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #16a085;
    }

    .result {
        color: #2c3e50;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
        background: rgba(255, 255, 255, 0.4);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }

    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load the trained model and tokenizer
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('gru_model110.h5')
    with open('tokenizer10.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model()

# Clean text function (same as in your notebook)
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict sentiment
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=300, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]

    if 0.42 <= prediction <= 0.62:
        sentiment = "Neutral"
        confidence = 1 - abs(prediction - 0.5) * 2  # Neutral confidence closer to 1 when nearer to 0.5
    elif prediction > 0.62:
        sentiment = "Positive"
        confidence = prediction
    else:
        sentiment = "Negative"
        confidence = 1 - prediction

    return sentiment, confidence


# Streamlit app
st.markdown('<div class="title">🎥Movie Review Sentiment Analysis App</div>', unsafe_allow_html=True)

# Input text area
user_input = st.text_area("Enter your review here:", height=200, placeholder="Type your movie review...")

# Button to predict
if st.button("Analyze Sentiment"):
    if user_input:
        sentiment, confidence = predict_sentiment(user_input)
        st.markdown(f'<div class="result">Sentiment: {sentiment} (Confidence: {confidence:.2%})</div>', unsafe_allow_html=True)

        # Visualization
        fig, ax = plt.subplots(figsize=(6, 4))
        sentiments = ['Negative', 'Positive']
        values = [1 - confidence, confidence] if sentiment == "Positive" else [confidence, 1 - confidence]
        ax.bar(sentiments, values, color=['#ff9999', '#99ff99'])
        ax.set_ylim(0, 1)
        ax.set_title("Sentiment Probability")
        ax.set_ylabel("Probability")
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.2%}', ha='center')
        st.pyplot(fig)
    else:
        st.warning("Please enter a review to analyze!")

# Footer
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; font-size: 12px; margin-top: 20px;'>
        Powered by Streamlit | Model trained on IMDB Dataset
    </div>
    """,
    unsafe_allow_html=True
)
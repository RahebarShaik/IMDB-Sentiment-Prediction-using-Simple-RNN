import numpy as np
import re
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

#Load the IMDB dataset
word_index = imdb.get_word_index()
rev_word_index = {value:key for key, value in word_index.items()}

#Load the Model
model = tf.keras.models.load_model('simple_rnn_model.h5')

#Function to encode the review
def pre_process(text):
    # 1. Strip punctuation so "bad!" becomes "bad"
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
        
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review

#Prediction Function
def predict_sentiment(text):
    processed_review = pre_process(text)
    prediction = model.predict(processed_review)[0][0]
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    return sentiment, prediction 

#Streamlit App
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

review_input = st.text_area('Movie Review')

if st.button('Predict Sentiment'):
    pre_processed_input = pre_process(review_input)
    prediction = model.predict(pre_processed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    #Display the result
    st.subheader("Prediction Result")
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {prediction[0][0]:.4f}")
else:
    st.write("Please enter a movie review and click 'Predict Sentiment' to see the result.")

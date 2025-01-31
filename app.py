import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load saved model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Preprocessing function
def preprocess_text(text):
    nltk.download('stopwords')
    nltk.download('wordnet')
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(text)

# Streamlit UI
st.title("Sentiment Analysis App")
st.subheader("Enter a review and predict if it's Positive or Negative")

user_input = st.text_area("Enter your review here:")

if st.button("Predict"):
    processed_text = preprocess_text(user_input)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(vectorized_text)[0]
    
    if prediction == 1:
        st.success("Positive Review 😊")
    else:
        st.error("Negative Review 😞")

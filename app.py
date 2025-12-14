import streamlit as st
import joblib
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Download NLTK data if not present
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load models
nb_model = joblib.load('models/nb_spam_model.joblib')
lr_model = joblib.load('models/lr_spam_model.joblib')
lstm_model = load_model('models/lstm_spam_model.h5')

# Load tokenizer for LSTM
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Preprocessing functions (same as in notebook)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)   # remove urls
    text = re.sub(r'[^a-z0-9\s]', ' ', text)      # keep alphanumerics
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Streamlit app
st.title("SMS Spam Detection Demo")
st.write("Enter an SMS message below to check if it's spam or ham using three different models.")

user_input = st.text_area("Enter SMS message:", height=100)

if st.button("Predict"):
    if user_input.strip():
        # Preprocess input
        cleaned = clean_text(user_input)
        processed = tokenize_and_lemmatize(cleaned)
        
        # Predictions
        nb_pred = nb_model.predict([processed])[0]
        nb_prob = nb_model.predict_proba([processed])[0][1]  # Probability of spam
        
        lr_pred = lr_model.predict([processed])[0]
        lr_prob = lr_model.predict_proba([processed])[0][1]
        
        # LSTM prediction
        seq = pad_sequences(tokenizer.texts_to_sequences([processed]), maxlen=100, truncating='post')
        lstm_prob = float(lstm_model.predict(seq)[0][0])
        lstm_pred = 1 if lstm_prob > 0.5 else 0
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Naive Bayes**")
            st.write(f"Prediction: {'Spam' if nb_pred == 1 else 'Ham'}")
            st.write(".3f")
        
        with col2:
            st.write("**Logistic Regression**")
            st.write(f"Prediction: {'Spam' if lr_pred == 1 else 'Ham'}")
            st.write(".3f")
        
        with col3:
            st.write("**LSTM**")
            st.write(f"Prediction: {'Spam' if lstm_pred == 1 else 'Ham'}")
            st.write(".3f")
        
        # Consensus
        preds = [nb_pred, lr_pred, lstm_pred]
        spam_count = sum(preds)
        if spam_count >= 2:
            consensus = "Spam (Majority)"
        else:
            consensus = "Ham (Majority)"
        st.write(f"**Consensus Prediction:** {consensus}")
    else:
        st.error("Please enter a message to predict.")
import streamlit as st
import joblib
import pickle
import re
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

# Page config
st.set_page_config(page_title="SMS Spam Detection", page_icon="📧", layout="wide")

# Fix SSL certificate issue for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data if not present (including punkt_tab for newer NLTK)
@st.cache_resource
def download_nltk_data():
    resources = {
        'tokenizers/punkt': 'punkt',
        'tokenizers/punkt_tab': 'punkt_tab',
        'corpora/wordnet': 'wordnet',
        'corpora/omw-1.4': 'omw-1.4',
        'corpora/stopwords': 'stopwords'
    }
    
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

download_nltk_data()

# Compatibility fix for loading old tokenizer
class CompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if 'keras.preprocessing' in module or 'keras.src.preprocessing' in module:
            if 'text' in module:
                module = 'keras_preprocessing.text'
            elif 'sequence' in module:
                module = 'keras_preprocessing.sequence'
        return super().find_class(module, name)

# Load models with caching for speed
@st.cache_resource
def load_models():
    nb = joblib.load('models/nb_spam_model.joblib')
    lr = joblib.load('models/lr_spam_model.joblib')
    lstm = load_model('models/lstm_spam_model.h5', compile=False)
    
    with open('models/tokenizer.pkl', 'rb') as f:
        tok = CompatibilityUnpickler(f).load()
    
    return nb, lr, lstm, tok

nb_model, lr_model, lstm_model, tokenizer = load_models()

# Initialize NLTK tools
@st.cache_resource
def get_nltk_tools():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return lemmatizer, stop_words

lemmatizer, stop_words = get_nltk_tools()

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
st.title("📧 SMS Spam Detection")
st.write("Detect spam messages using machine learning models")

# Sidebar for model selection
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.radio(
    "Select Model:",
    ["All Models (Consensus)", "Naive Bayes", "Logistic Regression", "LSTM Neural Network"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About Models
- **Naive Bayes**: Fast probabilistic classifier
- **Logistic Regression**: Linear model with high accuracy
- **LSTM**: Deep learning for sequence patterns
- **Consensus**: Majority vote from all three
""")

# Main input area
user_input = st.text_area(
    "Enter SMS message:",
    height=120,
    placeholder="Type or paste your SMS message here..."
)

# Example messages
with st.expander("📝 Try Example Messages"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Example: Ham Message"):
            st.session_state.example_text = "Hey, are we still on for dinner tonight at 7pm?"
    with col2:
        if st.button("Example: Spam Message"):
            st.session_state.example_text = "URGENT! You have won $1000! Call now to claim your prize: 555-0123"

if 'example_text' in st.session_state:
    user_input = st.session_state.example_text
    del st.session_state.example_text
    st.rerun()

if st.button("🔍 Predict", type="primary", use_container_width=True):
    if user_input.strip():
        with st.spinner("Analyzing message..."):
            # Preprocess input
            cleaned = clean_text(user_input)
            processed = tokenize_and_lemmatize(cleaned)
            
            # Display results based on selected model
            if model_choice == "All Models (Consensus)":
                # Get all predictions
                nb_pred = nb_model.predict([processed])[0]
                nb_prob = nb_model.predict_proba([processed])[0][1]
                
                lr_pred = lr_model.predict([processed])[0]
                lr_prob = lr_model.predict_proba([processed])[0][1]
                
                seq = pad_sequences(tokenizer.texts_to_sequences([processed]), maxlen=100, truncating='post')
                lstm_prob = float(lstm_model.predict(seq, verbose=0)[0][0])
                lstm_pred = 1 if lstm_prob > 0.5 else 0
                
                # Display results
                st.subheader("📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Naive Bayes",
                        "🚫 SPAM" if nb_pred == 1 else "✅ HAM",
                        f"{nb_prob:.1%} confidence"
                    )
                
                with col2:
                    st.metric(
                        "Logistic Regression",
                        "🚫 SPAM" if lr_pred == 1 else "✅ HAM",
                        f"{lr_prob:.1%} confidence"
                    )
                
                with col3:
                    st.metric(
                        "LSTM",
                        "🚫 SPAM" if lstm_pred == 1 else "✅ HAM",
                        f"{lstm_prob:.1%} confidence"
                    )
                
                # Consensus
                preds = [nb_pred, lr_pred, lstm_pred]
                spam_count = sum(preds)
                
                st.markdown("---")
                if spam_count >= 2:
                    st.error("### 🚫 CONSENSUS: SPAM")
                    st.write(f"{spam_count} out of 3 models classified this as spam")
                else:
                    st.success("### ✅ CONSENSUS: HAM (Not Spam)")
                    st.write(f"{3-spam_count} out of 3 models classified this as legitimate")
                    
            elif model_choice == "Naive Bayes":
                nb_pred = nb_model.predict([processed])[0]
                nb_prob = nb_model.predict_proba([processed])[0][1]
                
                st.subheader("📊 Naive Bayes Result")
                if nb_pred == 1:
                    st.error(f"### 🚫 SPAM (Confidence: {nb_prob:.1%})")
                else:
                    st.success(f"### ✅ HAM - Not Spam (Confidence: {(1-nb_prob):.1%})")
                    
            elif model_choice == "Logistic Regression":
                lr_pred = lr_model.predict([processed])[0]
                lr_prob = lr_model.predict_proba([processed])[0][1]
                
                st.subheader("📊 Logistic Regression Result")
                if lr_pred == 1:
                    st.error(f"### 🚫 SPAM (Confidence: {lr_prob:.1%})")
                else:
                    st.success(f"### ✅ HAM - Not Spam (Confidence: {(1-lr_prob):.1%})")
                    
            else:  # LSTM
                seq = pad_sequences(tokenizer.texts_to_sequences([processed]), maxlen=100, truncating='post')
                lstm_prob = float(lstm_model.predict(seq, verbose=0)[0][0])
                lstm_pred = 1 if lstm_prob > 0.5 else 0
                
                st.subheader("📊 LSTM Neural Network Result")
                if lstm_pred == 1:
                    st.error(f"### 🚫 SPAM (Confidence: {lstm_prob:.1%})")
                else:
                    st.success(f"### ✅ HAM - Not Spam (Confidence: {(1-lstm_prob):.1%})")
            
            # Show preprocessed text in expander
            with st.expander("🔍 View Preprocessed Text"):
                st.code(processed)
                
    else:
        st.error("⚠️ Please enter a message to predict.")
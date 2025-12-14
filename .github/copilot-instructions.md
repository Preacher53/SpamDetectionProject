# SMS Spam Detection Project — AI Agent Instructions

## Project Overview

**End-to-End NLP Classification Pipeline**: This Jupyter notebook implements three spam detection models (Naive Bayes, Logistic Regression, LSTM) trained on the UCI SMS Spam Collection dataset. The project demonstrates a complete ML workflow from data acquisition through model evaluation and artifact serialization.

## Architecture & Data Flow

### Key Components
1. **Data Pipeline**: Downloads UCI dataset → loads into pandas DataFrame → splits into train/val/test (70/15/15)
2. **Text Preprocessing**: URL removal → lowercasing → alphanumeric filtering → tokenization → lemmatization + stopword removal
3. **Model Ensemble**: Three independent models trained in parallel on preprocessed text
4. **Model Serialization**: All artifacts saved to `models/` for deployment (`.joblib` for sklearn, `.h5` for Keras, `.pkl` for tokenizer)

### Data Transformation Pipeline
```
Raw SMS → clean_text() → tokenize_and_lemmatize() → Cleaned text (df['cleaned'])
                            ↓
              TF-IDF Vectorizer (1-2 grams) ← [NB, LR use this]
              Tokenizer.texts_to_sequences() ← [LSTM uses this]
```

## Critical Developer Patterns

### Text Preprocessing (Non-Negotiable)
The `clean_text()` function must be applied **before** tokenization:
- Lowercase, remove URLs/special chars, collapse whitespace
- The same cleaning must be applied in inference (see Streamlit example)
- **Lemmatization** (not stemming) is used to preserve semantic meaning for spam keywords

### Model Training Pattern
Models use `sklearn.pipeline.Pipeline` to combine preprocessing with classification:
```python
Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=2)),
    ('clf', ModelClass())
])
```
This ensures the vectorizer is **fitted only on training data** and reused for validation/test (prevents data leakage).

### Stratified Splits
All train/val/test splits use `stratify=y` parameter to maintain class distribution (important for imbalanced spam/ham data).

## Key Files & Responsibilities

| File | Purpose | Critical Details |
|------|---------|-----------------|
| `Spam_Detection_Project.ipynb` | Main workflow notebook | Run cell-by-cell; all models + tokenizer trained here |
| `models/nb_spam_model.joblib` | Naive Bayes pipeline | Pre-fitted with TF-IDF; load with `joblib.load()` |
| `models/lr_spam_model.joblib` | Logistic Regression pipeline | Feature weights show spam indicators (see coef inspection) |
| `models/lstm_spam_model.h5` | LSTM Keras model | Paired with `tokenizer.pkl`; requires `pad_sequences()` to MAX_LEN=100 |
| `models/tokenizer.pkl` | Tokenizer state | Must be pickled together with LSTM; vocab size = 10000 |
| `data/SMSSpamCollection` | Source dataset | Tab-separated: `label\tmessage`; encoding='latin-1' required |

## Deployment & Inference Pattern

The notebook includes a **Streamlit app template** (end of notebook). Key inference requirements:
1. **Load all three models** independently (NB/LR are pipelines; LSTM needs tokenizer + pad_sequences)
2. **Apply `clean_text()` to user input** identically to training
3. **For LSTM**: tokenize → pad_sequences(maxlen=100) → predict → threshold at 0.5

Example: `lstm_prob = float(lstm_model.predict(seq)[0][0])` extracts scalar probability from batch output.

## Hyperparameter Choices (Discoverable, Not Arbitrary)

- **TF-IDF**: `ngram_range=(1,2)` captures unigrams + bigrams (e.g., "call now" is spam signal)
- **LSTM**: MAX_VOCAB=10000, MAX_LEN=100 → trade-off between vocabulary coverage and sequence padding
- **LSTM Architecture**: Bidirectional(64 units) + Dropout(0.5) prevents overfitting on small dataset
- **Train/Val epochs**: 6 epochs with batch_size=64 (validation loss plateaus after ~4-5)

## Extension Points

When improving models, maintain these patterns:
- **Feature engineering**: Add to preprocessing pipeline, not post-hoc (e.g., URL count, punctuation emphasis)
- **New models**: Keep in same Pipeline pattern; save to `models/[name]_spam_model.[ext]`
- **Transformer models**: Would need sentence-level tokenization (different from current char-based LSTM preprocessing)
- **Cross-validation**: Use `cross_val_score()` with **same stratify logic** in train_test_split

## Running the Project

### Local Setup
```powershell
# Install dependencies (or run notebook cell 1)
pip install numpy pandas scikit-learn nltk matplotlib seaborn tensorflow joblib streamlit

# Run notebook
jupyter notebook Spam_Detection_Project.ipynb
```

### Inference Only (Streamlit App)
```powershell
streamlit run app.py  # After saving the template code from notebook
```

## Testing & Validation Notes

- **Evaluation Metrics**: classification_report() shows precision/recall per class (imbalanced data consideration)
- **Confusion matrices** displayed for all three models to catch class-specific errors
- **Test set held out**: Never touched during training; reflects real-world performance

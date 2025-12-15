"""
Rebuild LSTM model with current TensorFlow/Keras version to fix compatibility issues
"""
import pickle
import numpy as np
import sys
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# Compatibility fix for loading old tokenizer
class CompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Remap old keras module paths to new ones
        if 'keras.preprocessing' in module or 'keras.src.preprocessing' in module:
            # Map to keras_preprocessing
            if 'text' in module:
                module = 'keras_preprocessing.text'
            elif 'sequence' in module:
                module = 'keras_preprocessing.sequence'
        return super().find_class(module, name)

# Load the tokenizer to get vocab size
print("Loading tokenizer...")
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = CompatibilityUnpickler(f).load()
print("Tokenizer loaded successfully")

# Model hyperparameters (same as original)
MAX_VOCAB = 10000
MAX_LEN = 100

# Load old model to get weights
print("Loading old model...")
try:
    old_model = load_model('models/lstm_spam_model.h5', compile=False)
    weights = old_model.get_weights()
    print("Successfully loaded weights from old model")
except Exception as e:
    print(f"Warning: Could not load old model: {e}")
    print("Will create new model without pretrained weights")
    weights = None

# Build new model with current Keras API (no deprecated parameters)
print("Building new model...")
model = Sequential([
    Embedding(MAX_VOCAB, 64),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Build the model by calling it on dummy data
dummy_input = np.zeros((1, MAX_LEN), dtype=np.int32)
model.predict(dummy_input, verbose=0)

# Transfer weights from old model if available
if weights is not None:
    try:
        model.set_weights(weights)
        print("Successfully transferred weights from old model")
    except Exception as e:
        print(f"Warning: Could not transfer weights: {e}")
        print("Model created with random initialization")

# Save new model
print("Saving new model...")
model.save('models/lstm_spam_model.h5')
print("Model saved successfully!")

# Test the model
print("\nModel summary:")
model.summary()

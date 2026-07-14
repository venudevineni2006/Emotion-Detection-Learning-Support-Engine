import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocessing import preprocess_text

# Load model
model = tf.keras.models.load_model(
    "models/bilstm/bilstm.keras"
)

# Load tokenizer
with open("models/bilstm/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("models/bilstm/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 80


def predict_bilstm(text):
    """
    Predict emotion using the BiLSTM model.
    Returns a unified prediction format.
    """

    # Preprocess
    cleaned = preprocess_text(text)

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned])

    # Pad
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # Predict
    prediction = model.predict(padded, verbose=0)[0]

    # Build probability dictionary
    probabilities = {}

    for emotion, probability in zip(
        label_encoder.classes_,
        prediction
    ):
        probabilities[emotion] = float(probability)

    # Primary emotion
    emotion = max(
        probabilities,
        key=probabilities.get
    )

    confidence = probabilities[emotion] * 100

    return {

        "emotion": emotion,

        "confidence": confidence,

        "probabilities": probabilities

    }
import pickle
import torch
import torch.nn.functional as F

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

MODEL_PATH = "models/bert_emotion_model_final"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

with open("models/bert_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


def predict_bert(text):
    """
    Predict emotion using the BERT model.
    Returns a unified prediction format.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=80
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probs = F.softmax(
            outputs.logits,
            dim=1
        )[0]

    probabilities = {}

    for emotion, probability in zip(
        label_encoder.classes_,
        probs
    ):
        probabilities[emotion] = float(probability)

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
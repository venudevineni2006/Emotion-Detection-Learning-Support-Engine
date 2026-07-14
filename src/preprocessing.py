import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Clean and preprocess input text.
    """

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(tokens)


EMOTION_RESPONSES = {

    "Confused": {
        "emoji": "🤔",
        "response": "I can see you're feeling confused. Don't worry—many students experience this when learning something new. Let's break the topic into smaller parts and understand it one step at a time.",
        "action": "Review the basics before moving to advanced concepts."
    },

    "Frustrated": {
        "emoji": "😣",
        "response": "I understand this feels frustrating. Take a short break, then revisit the problem with a fresh mind. Solving one small part at a time often makes the whole topic easier.",
        "action": "Try an alternative explanation or watch a beginner-friendly tutorial."
    },

    "Confident": {
        "emoji": "😎",
        "response": "Great job! You're showing confidence in this topic. This is the perfect time to challenge yourself with more advanced problems.",
        "action": "Attempt higher-level practice questions."
    },

    "Bored": {
        "emoji": "🥱",
        "response": "Learning doesn't have to be boring. Let's make it more engaging with visual explanations, coding exercises, quizzes, or interactive examples.",
        "action": "Try interactive exercises or real-world applications."
    },

    "Curious": {
        "emoji": "🧐",
        "response": "Curiosity is one of the best ways to learn. Since you're interested, let's explore the topic in more depth and connect it with practical applications.",
        "action": "Read advanced articles or explore related topics."
    }

}
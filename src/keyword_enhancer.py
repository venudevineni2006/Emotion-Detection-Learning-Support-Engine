# src/keyword_enhancer.py

emotion_keywords = {
    "Frustrated": [
        "stuck", "can't", "cannot", "difficult", "hard",
        "frustrated", "failed", "failure", "stress",
        "confusing", "hate", "problem", "error"
    ],

    "Confused": [
        "confused", "understand", "why", "how",
        "doubt", "unclear", "lost", "unsure"
    ],

    "Curious": [
        "curious", "explore", "learn", "interested",
        "discover", "wonder", "question"
    ],

    "Confident": [
        "easy", "understood", "confident", "know",
        "solved", "completed", "finished", "success"
    ],

    "Bored": [
        "boring", "bored", "sleepy", "tired",
        "lazy", "uninterested", "dull"
    ]
}


def boost_prediction(text, probabilities, boost=0.10):
    """
    Boosts emotion probabilities based on keywords.
    """

    text = text.lower()

    for emotion, words in emotion_keywords.items():

        for word in words:

            if word in text:

                probabilities[emotion] += boost

    # Normalize probabilities
    total = sum(probabilities.values())

    if total > 0:
        probabilities = {
            key: value / total
            for key, value in probabilities.items()
        }

    return probabilities
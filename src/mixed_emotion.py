def detect_mixed_emotions(predictions, threshold=0.15):
    """
    Detect primary and secondary emotions.

    Parameters:
        predictions (dict): Emotion -> Probability
        threshold (float): Minimum probability for secondary emotion

    Returns:
        dict
    """

    # Sort emotions by probability
    sorted_predictions = sorted(
        predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    primary_emotion = sorted_predictions[0][0]
    primary_score = sorted_predictions[0][1]

    secondary_emotions = []

    for emotion, score in sorted_predictions[1:]:

        if score >= threshold:
            secondary_emotions.append({
                "emotion": emotion,
                "score": round(score, 4)
            })

    return {
        "primary_emotion": primary_emotion,
        "primary_score": round(primary_score, 4),
        "secondary_emotions": secondary_emotions
    }
def get_mixed_emotions(scores, threshold=0.15):

    mixed = []

    for emotion, score in scores.items():

        if score >= threshold:

            mixed.append((emotion, score))

    mixed.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return mixed
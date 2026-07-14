from datetime import datetime


def create_prediction(
    input_text,
    bilstm_prediction,
    bert_prediction,
    mixed_result
):
    """
    Create a unified prediction schema.
    """

    return {

        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "input_text": input_text,

        "bilstm_prediction": bilstm_prediction,

        "bert_prediction": bert_prediction,

        "primary_emotion": mixed_result["primary_emotion"],

        "primary_score": mixed_result["primary_score"],

        "secondary_emotions": mixed_result["secondary_emotions"]

    }

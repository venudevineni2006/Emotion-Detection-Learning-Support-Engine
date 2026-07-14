import os
import pandas as pd
from datetime import datetime


CSV_FILE = "emotion_response_examples.csv"


def save_to_csv(field, problem, emotion, confidence, ai_response):

    try:

        row = {
            "field": field,
            "problem": problem,
            "emotion": emotion,
            "confidence": confidence,
            "response": ai_response,
            "timestamp": datetime.now().isoformat()
        }

        if os.path.exists(CSV_FILE):

            df = pd.read_csv(CSV_FILE)

            df = pd.concat(
                [df, pd.DataFrame([row])],
                ignore_index=True
            )

        else:

            df = pd.DataFrame([row])

        df.to_csv(
            CSV_FILE,
            index=False
        )

        return True

    except Exception as e:

        print(e)

        return False
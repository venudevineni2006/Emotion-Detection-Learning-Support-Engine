import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model_gemini = genai.GenerativeModel("gemini-2.5-flash")


def build_prompt(field, problem, emotion, confidence):
    return f"""
You are a helpful learning assistant.

A student studying {field} is feeling {emotion}
(confidence: {confidence:.2f}%).

Problem:
{problem}

Please provide:

1. Acknowledge the student's emotion.
2. Give one field-specific study tip.
3. Suggest one next step.
4. Encourage the student.

Use simple English.
Do not use markdown.
"""


def get_gemini_response(field, problem, emotion, confidence):
    try:

        prompt = build_prompt(
            field,
            problem,
            emotion,
            confidence
        )

        response = model_gemini.generate_content(prompt)

        return response.text.strip()

    except Exception as e:

        print("Gemini Error:", e)

        return None
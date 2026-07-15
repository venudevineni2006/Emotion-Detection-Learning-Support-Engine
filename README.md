# 🎓 Emotion Detection & Learning Support Engine

An AI-powered web application that detects a student's emotional state from their learning-related text and provides personalized learning support using Deep Learning and Generative AI.

---

## 📌 Project Overview

Learning can be challenging, and students often experience emotions such as confusion, frustration, boredom, curiosity, or confidence while studying. Traditional learning platforms provide the same content to every learner without considering their emotional state.

The **Emotion Detection & Learning Support Engine** addresses this problem by identifying a student's emotion from textual input and providing personalized, emotion-aware learning guidance.

The application combines Deep Learning models (BiLSTM and Fine-Tuned BERT) with Google's Gemini AI to generate supportive and field-specific responses that help students continue learning effectively.

---

# ✨ Features

- 😊 Emotion Detection from student text
- 🧠 BiLSTM Emotion Classification
- 🤖 Fine-Tuned BERT Emotion Classification
- 🔄 Model Comparison (BiLSTM vs BERT)
- 🎭 Mixed Emotion Detection
- 📊 Emotion Confidence Visualization
- 📈 Analytics Dashboard
- 📝 CSV Prediction Logging
- 💡 AI-Powered Learning Guidance using Gemini AI
- 📚 Personalized Learning Suggestions
- ⚡ Responsive Streamlit Interface
- 💾 Session State Management

---

# 🎯 Problem Statement

Students often struggle to understand difficult concepts and experience emotions such as confusion, frustration, boredom, or lack of confidence.

Existing learning platforms mainly focus on delivering educational content and do not recognize the learner's emotional state. As a result, students may lose motivation and become less engaged.

This project aims to bridge this gap by combining emotion detection with personalized AI-generated learning support.

---

# 🧠 Emotion Classes

The system classifies student emotions into five categories:

- 🤔 Confused
- 😣 Frustrated
- 😎 Confident
- 🥱 Bored
- 🧐 Curious

---

# 🏗 Project Architecture

```
Student Input
       │
       ▼
Text Preprocessing
       │
       ▼
────────────────────────
│                      │
▼                      ▼
BiLSTM Model       Fine-Tuned BERT
│                      │
└──────────┬───────────┘
           ▼
Emotion Prediction
           ▼
Mixed Emotion Detection
           ▼
Gemini AI Response Generator
           ▼
Learning Support
           ▼
Analytics & CSV Logging
```

---

# 📂 Project Structure

```
emotion-detection/
│
├── app.py
├── requirements.txt
├── .gitignore
├── .env
│
├── data/
│   └── predictions.csv
│
├── models/
│   ├── bilstm/
│   └── bert_emotion_model_final/
│
├── src/
│   ├── preprocessing.py
│   ├── bilstm_predict.py
│   ├── bert_predict.py
│   ├── gemini_helper.py
│   ├── logger.py
│   ├── mixed_emotion.py
│   ├── prediction_schema.py
│   ├── keyword_enhancer.py
│   └── utils.py
│
└── emotion_response_examples.csv
```

---

# 🛠 Technologies Used

## Programming Language

- Python

## Framework

- Streamlit

## Deep Learning

- TensorFlow
- Keras
- Transformers

## NLP

- NLTK
- Hugging Face Transformers

## Machine Learning

- Scikit-Learn

## Data Processing

- NumPy
- Pandas

## Visualization

- Plotly

## Generative AI

- Google Gemini API

---

# 📊 Models Used

## BiLSTM

- Embedding Layer
- Bidirectional LSTM
- Dense Layers
- Softmax Output

Used for fast emotion prediction.

---

## Fine-Tuned BERT

Pretrained BERT model fine-tuned on an emotion dataset for improved contextual understanding.

---

# 📈 Workflow

1. Student enters a learning-related problem.
2. Input text is preprocessed.
3. BiLSTM predicts the primary emotion.
4. Fine-Tuned BERT predicts the emotion independently.
5. Both models are compared.
6. Mixed emotions are detected.
7. Gemini AI generates a personalized learning response.
8. Prediction is logged to CSV.
9. Analytics dashboard is updated.

---

# 📊 Dashboard Features

- Emotion Prediction
- Confidence Scores
- Mixed Emotion Detection
- Model Comparison
- Analytics Dashboard
- CSV Logging
- Session History

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/venudevineni2006/Emotion-Detection-Learning-Support-Engine.git
```

Go to the project directory

```bash
cd Emotion-Detection-Learning-Support-Engine
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🔑 Environment Variables

Create a `.env` file inside the project root.

```
GEMINI_API_KEY=YOUR_API_KEY
```

---

# 📸 Screenshots

Add screenshots of:

- Home Page
- Emotion Detection
- Model Comparison
- Analytics Dashboard
- AI Learning Assistant

---

# 🚀 Future Enhancements

- Voice-Based Emotion Detection
- Speech Emotion Recognition
- Multilingual Emotion Detection
- Adaptive Learning Paths
- Learning Management System Integration
- Instructor Dashboard
- Mobile Application
- Real-Time Learning Analytics

---

# 📄 License

This project is developed for educational and academic purposes.

---

# 👨‍💻 Author

**Venu Gopal**

AI & Machine Learning Student

Project Title:

**Emotion Detection & Learning Support Engine**

GitHub:

https://github.com/venudevineni2006

---

# 🙏 Acknowledgements

- TensorFlow
- Hugging Face Transformers
- Google Gemini API
- Streamlit
- NLTK
- Plotly
- Scikit-Learn

---

## ⭐ If you find this project useful, consider giving it a star on GitHub.

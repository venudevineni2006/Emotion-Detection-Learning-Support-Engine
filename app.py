import streamlit as st
from src.bert_predict import predict_bert
from src.bilstm_predict import predict_bilstm
import src.gemini_helper as gh
from src.preprocessing import EMOTION_RESPONSES
from src.bilstm_predict import predict_bilstm
from src.preprocessing import EMOTION_RESPONSES
from src.logger import save_to_csv
from src.utils import get_mixed_emotions
import pandas as pd
import plotly.express as px

@st.cache_data
def create_dataframe(history):

    return pd.DataFrame(history)

st.set_page_config(
    page_title="Emotion Detection & Learning Support",
    page_icon="🎓",
    layout="wide"
)
# ----------------------------
# Session State Initialization
# ----------------------------

if "emotion_history" not in st.session_state:
    st.session_state["emotion_history"] = []

if "field" not in st.session_state:
    st.session_state["field"] = ""

if "problem" not in st.session_state:
    st.session_state["problem"] = ""

if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""

if "confidence" not in st.session_state:
    st.session_state["confidence"] = 0.0

if "ai_response" not in st.session_state:
    st.session_state["ai_response"] = ""
# ----------------------------
# Session State Initialization
# ----------------------------

with st.sidebar:

    st.header("📊 Dashboard")

    st.write("Models: ✅ Models Loaded")

    st.write(f"Total Interactions: {len(st.session_state['emotion_history'])}")
    ### waste###
    import os

    csv_file = "emotion_responses.csv"

    if os.path.exists(csv_file):

        df_csv = pd.read_csv(csv_file)

        st.write(f"CSV Examples: {len(df_csv)}")

    else:

        st.write("CSV Examples: 0")
    
    ###waste###

    if st.button("🗑️ Clear History"):

        st.session_state["emotion_history"] = []

        st.rerun()

    if st.session_state["emotion_history"]:

        st.subheader("Recent Sessions")

        recent = st.session_state["emotion_history"][-3:]

        for item in reversed(recent):

            st.write(
                f"• {item['field']} | {item['emotion']} | {item['confidence']:.1f}%"
            )




st.title("🎓 Emotion Detection & Learning Support Platform")
col1, col2 = st.columns([2, 1])
st.write(
    "Describe your learning problem and we'll analyze your emotional state."
)

st.sidebar.header("⚙ Settings")

use_ai = st.sidebar.checkbox(
    "Use AI Response (Gemini)",
    value=True
)

save_data = st.sidebar.checkbox(
    "Save to CSV",
    value=True
)

show_details = st.sidebar.checkbox(
    "Show analysis details",
    value=False
)
# --------------------------------------------------
# Field Selection
# --------------------------------------------------

with col1:

    st.subheader("📖 Tell us about your learning challenge")

    field = st.selectbox(
        "What are you studying?",
        [
            "Computer Science",
            "Mathematics",
            "Physics",
            "Chemistry",
            "Biology",
            "Engineering",
            "Business",
            "Literature",
            "History",
            "Psychology",
            "Other"
        ],
        help="Select your area of study."
    )

    problem = st.text_area(
        f"Describe your {field} problem:",
        placeholder=f"Example: I'm struggling with recursion in {field}.",
        height=120
    )

    st.write("Quick Examples")

    c1, c2, c3 = st.columns(3)

    with c1:

        if st.button("I'm confused about recursion"):

            problem = "I'm confused about recursion."

    with c2:

        if st.button("Debugging is frustrating"):

            problem = "Debugging is frustrating."

    with c3:

        if st.button("I'm curious about machine learning"):

            problem = "I'm curious about machine learning."


# --------------------------------------------------
# Analyze Button
# --------------------------------------------------

analyze = st.button(
    "Get AI Learning Help",
    use_container_width=True,
    type="primary",
    disabled=problem.strip() == ""
)

if analyze:

    if len(problem.strip()) < 10:

        st.error(
            "Please describe your learning problem in at least 10 characters."
        )

        st.stop()

    else:

        with st.spinner("Analyzing your learning state..."):

            # -------------------------
            # BiLSTM Prediction
            # -------------------------

            try:

                bilstm_result = predict_bilstm(problem)

                bert_result = predict_bert(problem)

            except Exception as e:

                st.error("Prediction failed.")

                st.exception(e)

                st.stop()

            emotion = bilstm_result["emotion"]

            confidence = bilstm_result["confidence"]

            probabilities = bilstm_result["probabilities"]
            
            if emotion == "":

                st.error("No emotion detected.")

                st.stop()
            st.session_state["field"] = field
            st.session_state["problem"] = problem
            st.session_state["emotion"] = emotion
            st.session_state["confidence"] = confidence
            st.session_state["probabilities"] = probabilities

            # -------------------------
            # Gemini Response
            # -------------------------

            
            try:

                ai_response = gh.get_gemini_response(
                                    field,
                                    problem,
                                    emotion,
                                    confidence
                              )

            except Exception:

                ai_response = None
            st.session_state["ai_response"] = ai_response

        st.success("Analysis Complete!")
        st.divider()
        st.subheader("🔥 Model Predictions Comparison")

        col1, col2 = st.columns(2)
        with col1:

            st.write("### 🧠 BiLSTM")

            bilstm_mixed = get_mixed_emotions(probabilities)

            if len(bilstm_mixed) > 1:

                mixed_text = ", ".join(

                    [
                        f"{emotion} ({score:.1%})"
                        for emotion, score in bilstm_mixed
                    ]

                )

                st.info(
                    f"Mixed Emotions\n\n{mixed_text}"
                )

            else:

                st.metric(
                    "Primary Emotion",
                    emotion,
                    f"{confidence:.1f}%"
                )

            st.write("Emotion Scores")

            for emotion_name, score in sorted(

                probabilities.items(),

                key=lambda x: x[1],

                reverse=True

            ):

                st.write(
                    f"**{emotion_name}** ({score:.1%})"
                )

                st.progress(score)
        with col2:

            st.write("### 🤖 BERT")

            bert_mixed = get_mixed_emotions(bert_result)

            bert_primary = max(

                bert_result,

                key=bert_result.get

            )

            bert_confidence = bert_result[bert_primary]

            if len(bert_mixed) > 1:

                mixed_text = ", ".join(

                    [

                        f"{emotion} ({score:.1%})"

                        for emotion, score in bert_mixed

                    ]

                )

                st.info(

                    f"Mixed Emotions\n\n{mixed_text}"

                )

            else:

                st.metric(

                    "Primary Emotion",

                    bert_primary,

                    f"{bert_confidence:.1%}"

                )

            st.write("Emotion Scores")

            for emotion_name, score in sorted(

                bert_result.items(),

                key=lambda x: x[1],

                reverse=True

            ):

                st.write(

                    f"**{emotion_name}** ({score:.1%})"

                )

                st.progress(score)

        # --------------------------------------------------
        # Emotion
        # --------------------------------------------------
        st.subheader("😊 Detected Emotion")

        st.write(emotion)

        # --------------------------------------------------
        # Confidence        
        # --------------------------------------------------

        st.subheader("📊 Confidence")

        st.write(f"{confidence:.2f}%")

        # --------------------------------------------------
        # Probabilities
        # --------------------------------------------------
        if show_details:

            st.subheader("Analysis Details")

            st.write("Original Problem")

            st.write(problem)

            st.write("BiLSTM Prediction")

            st.write(emotion)

            st.write("Confidence")

            st.write(f"{confidence:.2f}%")

            st.write("AI Model")

            st.write("Gemini 2.5 Flash")
        

        # --------------------------------------------------
        # AI Response / Fallback
        # --------------------------------------------------

    if ai_response:

        final_response = ai_response

    else:

        fallback = EMOTION_RESPONSES[emotion]

        final_response = fallback["response"]
    history_item = {

        "field": field,

        "problem": problem,

        "emotion": emotion,

        "confidence": confidence,

        "response": final_response

    }

    st.session_state["emotion_history"].append(history_item)

    if save_data:

        save_to_csv(

            field,

            problem,

            emotion,

            confidence,

            final_response

        )

    if ai_response:

      st.subheader("🤖 AI Learning Assistant")

      st.info(st.session_state["ai_response"])
      st.subheader("📚 Additional Support")

      st.info(f"Strategy: {EMOTION_RESPONSES[emotion]['action']}")

    # Regenerate Response Button
      if st.button(
          "🔄 Regenerate Response",
           use_container_width=True
      ):

            st.session_state["ai_response"] = gh.get_gemini_response(
                st.session_state["field"],
                st.session_state["problem"],
                st.session_state["emotion"],
                st.session_state["confidence"]
    
            )

            st.rerun()

    else:

        st.subheader(
           f"{fallback['emoji']} Learning Support"
        )

        st.write(fallback["response"])

        st.info(
           f"Recommended Action: {fallback['action']}"
        )
# ======================================================
# Analytics Dashboard
# ======================================================

if st.session_state["emotion_history"]:

    st.divider()

    st.header("📊 Learning Analytics")

    df = create_dataframe(st.session_state["emotion_history"])

    tab1, tab2, tab3 = st.tabs(
        [
            "😊 Emotions",
            "📚 Fields",
            "📈 Summary"
        ]
    )

    # -------------------------
    # Emotion Tab
    # -------------------------

    with tab1:

        col1, col2 = st.columns(2)

        with col1:

            emotion_counts = df["emotion"].value_counts()

            fig1 = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="Emotion Distribution"
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )
        
        with col2:

            df_copy = df.copy()

            df_copy["Session"] = range(
                1,
                len(df_copy) + 1
            )

            fig2 = px.line(
                df_copy,
                x="Session",
                y="confidence",
                color="emotion",
                markers=True,
                title="Emotional Journey"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )
    # -------------------------
    # Fields Tab
    # -------------------------

    with tab2:

        field_emotion = (df.groupby(["field", "emotion"]).size().reset_index(name="Count"))

        fig3 = px.bar(

            field_emotion,

            x="field",

            y="Count",

            color="emotion",

            barmode="group",

            title="Emotion Distribution Across Fields"

        )

        st.plotly_chart(

                fig3,

                use_container_width=True

        )

    # -------------------------
    # Summary Tab
    # -------------------------

    with tab3:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Sessions",
                len(df)
            )

        with col2:

            st.metric(
                "Average Confidence",
                f"{df['confidence'].mean():.1f}%"
            )

        with col3:

            st.metric(
                "Most Common Emotion",
                df["emotion"].mode()[0]
            )

        st.divider()

        st.subheader("History")

        st.dataframe(
            df,
            use_container_width=True
        )
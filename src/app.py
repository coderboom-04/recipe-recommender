import streamlit as st
import pandas as pd
from preprocess import load_and_clean_data
from model import RecipeRecommender
from audiorecorder import audiorecorder
import speech_recognition as sr


# Page settings
st.set_page_config(page_title="Smart Recipe Recommender 🍳", page_icon="🍳")

st.title("🍳 Smart Recipe Recommender")
st.write("Enter the ingredients you have, and get recipes you can cook!")


# Google Drive link (DIRECT DOWNLOAD)
DATA_URL = "https://drive.google.com/uc?export=download&id=18l8zaNZKUM2VgKNXfSKLAzh6GPyKGQcz"


@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None})
def load_model():
    df = load_and_clean_data(DATA_URL)
    model = RecipeRecommender(df)
    return model


model = load_model()


# -----------------------------------
# Voice Input
# -----------------------------------
st.subheader("🎤 Voice Input")
audio = audiorecorder("Start Recording 🎙️", "Stop Recording 🛑")

user_ingredients = ""

if len(audio) > 0:
    st.audio(audio.tobytes(), format="audio/wav")
    audio.export("voice_input.wav", format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile("voice_input.wav") as source:
        audio_data = recognizer.record(source)

    try:
        spoken_text = recognizer.recognize_google(audio_data)
        st.success(f"🗣️ You said: **{spoken_text}**")
        user_ingredients = spoken_text
    except:
        st.error("Could not understand your voice!")


# -----------------------------------
# Text Input
# -----------------------------------
text_input = st.text_input(
    "Enter ingredients (comma separated):",
    value=user_ingredients,
    placeholder="Example: egg, tomato, onion"
)

if text_input:
    st.subheader("🔍 Recommended Recipes")
    results = model.recommend_semantic(text_input, top_n=5)

    max_time = st.slider("⏱️ Maximum cooking time (minutes):", 1, 500, 60)

    results = results[results["minutes"] <= max_time]

    if results.empty:
        st.warning("No recipes match your time preference!")
    else:
        for _, row in results.iterrows():
            st.markdown(f"### 🍽️ {row['title']} — **{row['semantic_score']}% Match**")

            st.write(f"- **Calories:** {row['calories']:.0f}")
            st.write(f"- **Protein:** {row['protein']:.1f} g")
            st.write(f"- **Carbs:** {row['carbs']:.1f} g")
            st.write(f"- **Sugar:** {row['sugar']:.1f} g")
            st.write(f"- **Sodium:** {row['sodium']:.1f} mg")

            st.markdown(f"**Ingredients:** {row['ingredients']}")

            st.markdown("**Steps to Prepare:**")
            for i, step in enumerate(row["steps"], start=1):
                st.markdown(f"{i}. {step}")

            st.markdown("---")

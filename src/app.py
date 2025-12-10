import streamlit as st
import pandas as pd
from preprocess import load_and_clean_data
from model import RecipeRecommender
from streamlit_mic_recorder import speech_to_text


# -----------------------------
# 🌟 Streamlit Page Settings
# -----------------------------
st.set_page_config(page_title="Smart Recipe Recommender 🍳", page_icon="🍳")

st.title("🍳 Smart Recipe Recommender")
st.write("Enter ingredients OR speak them, and get recipes instantly!")


# -----------------------------
# 🌟 Load Model (Cached)
# -----------------------------
@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None})
def load_model():
    df = load_and_clean_data("RAW_recipes.csv")
    model = RecipeRecommender(df)
    return model

model = load_model()


# -----------------------------
# 🎤 Voice Input Section
# -----------------------------
st.subheader("🎤 Voice Input")

spoken_text = speech_to_text(language="en", key="voice_input")

if spoken_text:
    st.success(f"🗣️ You said: **{spoken_text}**")
    user_ingredients = spoken_text   # autofill ingredients
else:
    user_ingredients = ""


# -----------------------------
# ✍️ Manual Text Input
# -----------------------------
user_ingredients = st.text_input(
    "Enter ingredients (comma separated):",
    value=user_ingredients,
    placeholder="Example: egg, tomato, onion"
)


# -----------------------------
# 🔍 Run Recommendations
# -----------------------------
if user_ingredients:
    st.subheader("🔍 Recommended Recipes:")

    results = model.recommend_semantic(user_ingredients, top_n=5)

    # Cooking time filter
    max_time = st.slider(
        "⏱️ Maximum cooking time (minutes):",
        min_value=1,
        max_value=500,
        value=60,
        step=5
    )

    results = results[results['minutes'] <= max_time]

    if results.empty:
        st.warning("No recipes match your criteria. Try increasing the cooking time!")
    else:
        for _, row in results.iterrows():

            # Title + match %
            st.markdown(
                f"### 🍽️ {row['title'].title()} — **{row['semantic_score']:.2f}% Match**"
            )

            # Nutrition Info
            st.markdown("### 🥗 Nutrition Info (per serving):")
            st.write(f"- **Calories:** {row['calories']:.0f} kcal")
            st.write(f"- **Protein:** {row['protein']:.1f} g")
            st.write(f"- **Carbs:** {row['carbs']:.1f} g")
            st.write(f"- **Sugar:** {row['sugar']:.1f} g")
            st.write(f"- **Sodium:** {row['sodium']:.1f} mg")

            # Ingredients
            st.markdown(f"**🧾 Ingredients:** {row['ingredients']}")

            # Steps
            st.markdown("**👩‍🍳 Steps to Prepare:**")
            steps = row.get('steps', [])

            if isinstance(steps, list) and steps:
                for i, step in enumerate(steps, start=1):
                    st.markdown(f"{i}. {step}")
            else:
                st.markdown("_No steps available for this recipe._")

            st.markdown("---")

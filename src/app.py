import streamlit as st
import pandas as pd
from preprocess import load_and_clean_data
from model import RecipeRecommender

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Smart Recipe Recommender 🍳", page_icon="🍳")

st.title("🍳 Smart Recipe Recommender")
st.write("Enter the ingredients you have, and get recipes you can cook!")

# -------------------------------
# LOAD MODEL + DATA (CACHED)
# -------------------------------
@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None})
def load_model():
    df = load_and_clean_data("https://drive.google.com/uc?export=download&id=18l8zaNZKUM2VgKNXfSKLAzh6GPyKGQcz")
    model = RecipeRecommender(df)
    return model

model = load_model()

# -------------------------------
# USER INPUT TEXT BOX
# -------------------------------
user_ingredients = st.text_input(
    "Enter ingredients (comma separated):",
    placeholder="Example: egg, tomato, onion"
)

# -------------------------------
# RUN RECOMMENDER ONLY IF INPUT EXISTS
# -------------------------------
if user_ingredients:
    st.subheader("🔍 Recommended Recipes")

    # Get semantic recommendations
    results = model.recommend_semantic(user_ingredients, top_n=5)

    # Time filter
    max_time = st.slider(
        "⏱️ Maximum cooking time (minutes):",
        min_value=1,
        max_value=500,
        value=60,
        step=5
    )

    # Filter recipes by time
    results = results[results['minutes'] <= max_time]

    if results.empty:
        st.warning("No recipes fit your time range. Try increasing the cooking time!")
    else:
        for _, row in results.iterrows():
            st.markdown(
                f"### 🍽️ {row['title'].title()} — **{row['semantic_score']:.2f}% Match**"
            )

            st.markdown("### 🥗 Nutrition Info (per serving):")
            st.write(f"- **Calories:** {row['calories']:.0f} kcal")
            st.write(f"- **Protein:** {row['protein']:.1f} g")
            st.write(f"- **Carbs:** {row['carbs']:.1f} g")
            st.write(f"- **Sugar:** {row['sugar']:.1f} g")
            st.write(f"- **Sodium:** {row['sodium']:.1f} mg")

            st.markdown(f"**🧾 Ingredients:** {row['ingredients']}")

            steps = row.get("steps", [])

            st.markdown("**👩‍🍳 Steps to Prepare:**")
            if isinstance(steps, list) and steps:
                for i, step in enumerate(steps, start=1):
                    st.markdown(f"{i}. {step}")
            else:
                st.markdown("_No steps available for this recipe._")

            st.markdown("---")

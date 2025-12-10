import streamlit as st
import pandas as pd
import requests

from preprocess import load_and_clean_data
from model import RecipeRecommender


# ---------------------------------------------
# DOWNLOAD CSV FROM GOOGLE DRIVE
# ---------------------------------------------
@st.cache_data
def download_csv():
    url = "https://drive.google.com/uc?export=download&id=18l8zaNZKUM2VgKNXfSKLAzh6GPyKGQcz"
    response = requests.get(url)

    csv_path = "recipes.csv"
    with open(csv_path, "wb") as file:
        file.write(response.content)

    return csv_path


# ---------------------------------------------
# LOAD MODEL AND DATA
# ---------------------------------------------
@st.cache_resource
def load_model():
    csv_path = download_csv()
    df = load_and_clean_data(csv_path)
    return RecipeRecommender(df)


# Load ONCE
model = load_model()


# ---------------------------------------------
# FRONTEND
# ---------------------------------------------
st.title("🔍 Smart Recipe Recommender")
st.write("Enter the ingredients you have, and get recipes instantly!")

user_ingredients = st.text_input(
    "Enter ingredients (comma separated):",
    placeholder="Example: egg, tomato, onion"
)

if user_ingredients:

    st.subheader("🔍 Recommended Recipes")

    results = model.recommend_semantic(user_ingredients, top_n=5)

    # Filter by cooking time
    max_time = st.slider(
        "⏱️ Maximum cooking time (minutes):",
        min_value=1,
        max_value=500,
        value=60,
        step=5
    )

    if "minutes" in results.columns:
        results = results[results["minutes"] <= max_time]

    if results.empty:
        st.warning("No recipes matched. Try adding more ingredients or increasing time.")
    else:
        for _, row in results.iterrows():

            st.markdown(f"### 🍽️ {row['title'].title()} — **{row.get('semantic_score',0)}% Match**")

            # Nutrition
            st.write(f"- **Calories:** {row.get('calories',0)} kcal")
            st.write(f"- **Protein:** {row.get('protein',0)} g")
            st.write(f"- **Carbs:** {row.get('carbs',0)} g")
            st.write(f"- **Sugar:** {row.get('sugar',0)} g")
            st.write(f"- **Sodium:** {row.get('sodium',0)} mg")

            # Ingredients
            st.markdown(f"**🧾 Ingredients:** {row.get('ingredients','N/A')}")

            # Steps
            steps = row.get("steps", [])
            st.markdown("**👩‍🍳 Steps to Prepare:**")

            if isinstance(steps, list) and len(steps) > 0:
                for i, step in enumerate(steps, start=1):
                    st.markdown(f"{i}. {step}")
            else:
                st.write("_No steps available._")

            st.markdown("---")

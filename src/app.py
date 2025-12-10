import streamlit as st
import pandas as pd
import requests

from preprocess import load_and_clean_data
from model import RecipeRecommender


# ---------------------------------------------
# DOWNLOAD CSV FROM GOOGLE DRIVE (SMALL DATASET)
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
# LOAD MODEL + DATA (CACHED)
# ---------------------------------------------
@st.cache_resource
def load_model():
    csv_path = download_csv()
    df = load_and_clean_data(csv_path)
    return RecipeRecommender(df)


# Load model ONCE when app starts
model = load_model()


# ---------------------------------------------
# STREAMLIT FRONTEND
# ---------------------------------------------
st.title("🔍 Smart Recipe Recommender")
st.write("Enter the ingredients you have, and get recipes instantly!")

# User input
user_ingredients = st.text_input(
    "Enter ingredients (comma separated):",
    placeholder="Example: egg, tomato, onion"
)

# Only process if user enters ingredients
if user_ingredients:

    st.subheader("🔍 Recommended Recipes")

    # Get recommendations
    results = model.recommend_semantic(user_ingredients, top_n=5)

    # Cooking time filter
    max_time = st.slider(
        "⏱️ Maximum cooking time (minutes):",
        min_value=1,
        max_value=500,
        value=60,
        step=5
    )

    results = results[results["minutes"] <= max_time]

    if results.empty:
        st.warning("No recipes matched. Try adding more ingredients or increasing time.")
    else:
        # Display results
        for _, row in results.iterrows():

            st.markdown(f"### 🍽️ {row['title'].title()} — **{row['semantic_score']}% Match**")

            st.write(f"- **Calories:** {row['calories']} kcal")
            st.write(f"- **Protein:** {row['protein']} g")
            st.write(f"- **Carbs:** {row['carbs']} g")
            st.write(f"- **Sugar:** {row['sugar']} g")
            st.write(f"- **Sodium:** {row['sodium']} mg")

            st.markdown(f"**🧾 Ingredients:** {row['ingredients']}")

            steps = row.get("steps", [])
            st.markdown("**👩‍🍳 Steps to Prepare:**")

            if isinstance(steps, list) and len(steps) > 0:
                for i, step in enumerate(steps, start=1):
                    st.markdown(f"{i}. {step}")
            else:
                st.write("_No steps available._")

            st.markdown("---")

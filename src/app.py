import streamlit as st
import pandas as pd

from preprocess import load_and_clean_data
from model import RecipeRecommender


# ---------------------------------------------
# LOAD DATASET DIRECTLY FROM REPO (NO DOWNLOAD)
# ---------------------------------------------
@st.cache_resource
def load_model():
    csv_path = "data/RAW_recipes.csv"   # <-- your uploaded dataset
    df = load_and_clean_data(csv_path)
    return RecipeRecommender(df)


# Load model ONCE at app startup
model = load_model()


# ---------------------------------------------
# STREAMLIT UI
# ---------------------------------------------
st.title("🍽️ Smart Recipe Recommender")
st.write("Enter the ingredients you have — get recipes instantly!")


# -------- User Input --------
user_ingredients = st.text_input(
    "Enter ingredients (comma separated):",
    placeholder="Example: egg, tomato, onion"
)


# -------- Recommendation Logic --------
if user_ingredients:

    st.subheader("🔍 Recommended Recipes")

    # Get top 5 results
    results = model.recommend_semantic(user_ingredients, top_n=5)

    # Cooking Time Filter
    max_time = st.slider(
        "⏱️ Maximum cooking time (minutes):",
        min_value=1,
        max_value=500,
        value=60,
        step=5
    )

    # Filter results
    if "minutes" in results.columns:
        results = results[results["minutes"] <= max_time]

    if results.empty:
        st.warning("No recipes matched. Try adding more ingredients or increasing cooking time.")
    else:
        # Display recipes beautifully
        for _, row in results.iterrows():

            st.markdown(f"### 🍳 {row['title'].title()} — **{row['semantic_score']}% Match**")

            st.write(f"- **Calories:** {row['calories']} kcal")
            st.write(f"- **Protein:** {row['protein']} g")
            st.write(f"- **Carbs:** {row['carbs']} g")
            st.write(f"- **Sugar:** {row['sugar']} g")
            st.write(f"- **Sodium:** {row['sodium']} mg")

            st.markdown(f"**🧾 Ingredients:** {row['ingredients']}")

            # Steps
            steps = row.get("steps", [])
            st.markdown("**👩‍🍳 Steps to Prepare:**")

            if isinstance(steps, list) and len(steps) > 0:
                for i, step in enumerate(steps, start=1):
                    st.markdown(f"{i}. {step}")
            else:
                st.write("_No steps available._")

            st.markdown("---")

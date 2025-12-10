import streamlit as st
import pandas as pd
import requests
from preprocess import load_and_clean_data
from model import RecipeRecommender

st.set_page_config(page_title="Smart Recipe Recommender 🍳", page_icon="🍳")

st.title("🍳 Smart Recipe Recommender")
st.write("Enter the ingredients you have, and get recipes instantly!")

# ---- DOWNLOAD CSV FROM GOOGLE DRIVE ----
@st.cache_resource
def download_csv():
    url = "https://drive.google.com/uc?export=download&id=18l8zaNZKUM2VgKNXfSKLAzh6GPyKGQcz"
    r = requests.get(url)
    open("RAW_recipes.csv", "wb").write(r.content)
    return "RAW_recipes.csv"

# ---- LOAD MODEL ----
@st.cache_data
def load_model():
    # TEMPORARY — ensure Streamlit works
    import pandas as pd
    
    df = pd.DataFrame({
        "cleaned_ingredients": [
            "chicken salt pepper",
            "milk sugar flour",
            "egg cheese butter"
        ],
        "name": ["Chicken Dish", "Cake", "Omelette"]
    })

    from model import RecipeRecommender
    return RecipeRecommender(df)


# ---- USER INPUT ----
user_ingredients = st.text_input(
    "Enter ingredients (comma separated):",
    placeholder="Example: egg, tomato, onion"
)

if user_ingredients:
    st.subheader("🔍 Recommended Recipes")

    results = model.recommend_semantic(user_ingredients, top_n=5)

    max_time = st.slider(
        "⏱️ Maximum cooking time (minutes):",
        min_value=1,
        max_value=500,
        value=60,
        step=5
    )

    results = results[results["minutes"] <= max_time]

    if results.empty:
        st.warning("No recipes found for your time range.")
    else:
        for _, row in results.iterrows():
            st.markdown(f"### 🍽️ {row['title'].title()} — **{row['semantic_score']:.2f}% Match**")

            st.markdown("### 🥗 Nutrition")
            st.write(f"- **Calories:** {row['calories']:.0f} kcal")
            st.write(f"- **Protein:** {row['protein']:.1f} g")
            st.write(f"- **Carbs:** {row['carbs']:.1f} g")
            st.write(f"- **Sugar:** {row['sugar']:.1f} g")
            st.write(f"- **Sodium:** {row['sodium']:.1f} mg")

            st.markdown(f"**🧾 Ingredients:** {row['ingredients']}")

            st.markdown("**👩‍🍳 Steps:**")
            steps = row.get("steps", [])
            if isinstance(steps, list) and steps:
                for i, step in enumerate(steps, 1):
                    st.markdown(f"{i}. {step}")
            else:
                st.markdown("_No steps available._")

            st.markdown("---")

import pandas as pd
import ast
import re

def clean_ingredients(text):
    """Clean and normalize ingredient text."""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_clean_data(path):
    df = pd.read_csv(path)

    # -----------------------------
    # FIX 1 — If nutrition is missing, create dummy nutrition
    # -----------------------------
    if "nutrition" not in df.columns:
        df["nutrition"] = ["[0,0,0,0,0,0,0]"] * len(df)

    # Parse nutrition list
    def parse_nutrition(x):
        try:
            values = ast.literal_eval(x)
            return values if isinstance(values, list) else [0]*7
        except:
            return [0]*7

    df["nutrition_parsed"] = df["nutrition"].apply(parse_nutrition)

    df["calories"] = df["nutrition_parsed"].apply(lambda x: x[0])
    df["total_fat"] = df["nutrition_parsed"].apply(lambda x: x[1])
    df["sugar"] = df["nutrition_parsed"].apply(lambda x: x[2])
    df["sodium"] = df["nutrition_parsed"].apply(lambda x: x[3])
    df["protein"] = df["nutrition_parsed"].apply(lambda x: x[4])
    df["sat_fat"] = df["nutrition_parsed"].apply(lambda x: x[5])
    df["carbs"] = df["nutrition_parsed"].apply(lambda x: x[6])

    # -----------------------------
    # FIX 2 — Ingredients column
    # -----------------------------
    if "ingredients" in df.columns:
        df["ingredients"] = df["ingredients"].apply(
            lambda x: " ".join(ast.literal_eval(x)) if isinstance(x, str) else ""
        )
    else:
        df["ingredients"] = ""

    df["cleaned_ingredients"] = df["ingredients"].apply(clean_ingredients)

    # -----------------------------
    # FIX 3 — Steps column
    # -----------------------------
    if "steps" in df.columns:
        df["steps"] = df["steps"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
    else:
        df["steps"] = [[] for _ in range(len(df))]

    df.dropna(subset=["cleaned_ingredients"], inplace=True)

    return df

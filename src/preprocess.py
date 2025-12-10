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

    # ---------- FIX 1: guarantee ingredients column ----------
    if "ingredients" not in df.columns:
        df["ingredients"] = ""   # fallback

    else:
        # convert list string → sentence
        df["ingredients"] = df["ingredients"].apply(
            lambda x: " ".join(ast.literal_eval(x)) if isinstance(x, str) else ""
        )

    # Create cleaned ingredient text
    df["cleaned_ingredients"] = df["ingredients"].apply(clean_ingredients)

    # ---------- FIX 2: ensure cleaned_ingredients is NEVER empty ----------
    df["cleaned_ingredients"].replace("", "ingredient", inplace=True)

    # ---------- FIX 3: steps column ----------
    if "steps" in df.columns:
        df["steps"] = df["steps"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
    else:
        df["steps"] = [[] for _ in range(len(df))]

    # ---------- FIX 4: nutrition column ----------
    if "nutrition" not in df.columns:
        df["nutrition"] = ["[0,0,0,0,0,0,0]"] * len(df)

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

    return df

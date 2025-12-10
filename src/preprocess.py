import pandas as pd
import ast
import re

def clean_ingredients(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else "ingredient"
    return "ingredient"

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # ---- INGREDIENTS COLUMN FIX ----
    if "ingredients" not in df.columns:
        df["ingredients"] = ""

    def fix_ingredient_list(x):
        if isinstance(x, str):
            try:
                lst = ast.literal_eval(x)
                if isinstance(lst, list):
                    return " ".join(lst)
            except:
                pass
        return ""

    df["ingredients"] = df["ingredients"].apply(fix_ingredient_list)
    df["cleaned_ingredients"] = df["ingredients"].apply(clean_ingredients)

    # If ANY cleaned ingredient is empty, replace it
    df["cleaned_ingredients"].replace("", "ingredient", inplace=True)

    # ---- STEPS FIX ----
    if "steps" not in df.columns:
        df["steps"] = [[] for _ in range(len(df))]
    else:
        df["steps"] = df["steps"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )

    # ---- NUTRITION FIX ----
    if "nutrition" not in df.columns:
        df["nutrition"] = ["[0,0,0,0,0,0,0]"] * len(df)

    def parse_nutrition(x):
        try:
            vals = ast.literal_eval(x)
            if isinstance(vals, list) and len(vals) >= 7:
                return vals
        except:
            pass
        return [0,0,0,0,0,0,0]

    df["nutrition_parsed"] = df["nutrition"].apply(parse_nutrition)

    df["calories"] = df["nutrition_parsed"].apply(lambda x: x[0])
    df["total_fat"] = df["nutrition_parsed"].apply(lambda x: x[1])
    df["sugar"] = df["nutrition_parsed"].apply(lambda x: x[2])
    df["sodium"] = df["nutrition_parsed"].apply(lambda x: x[3])
    df["protein"] = df["nutrition_parsed"].apply(lambda x: x[4])
    df["sat_fat"] = df["nutrition_parsed"].apply(lambda x: x[5])
    df["carbs"] = df["nutrition_parsed"].apply(lambda x: x[6])

    return df

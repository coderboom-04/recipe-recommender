import pandas as pd
import ast
import re
import requests
from io import StringIO

def clean_ingredients(text):
    """Clean and normalize ingredient text."""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_clean_data(path_or_url):
    # If the input is a URL → download from Google Drive
    if path_or_url.startswith("http"):
        response = requests.get(path_or_url)
        response.raise_for_status()  # throw error if download fails
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
    else:
        df = pd.read_csv(path_or_url)

    # Parse nutrition list
    def parse_nutrition(x):
        try:
            values = ast.literal_eval(x)
            return values if isinstance(values, list) else [0] * 7
        except:
            return [0] * 7

    df['nutrition_parsed'] = df['nutrition'].apply(parse_nutrition)

    df['calories'] = df['nutrition_parsed'].apply(lambda x: x[0])
    df['total_fat'] = df['nutrition_parsed'].apply(lambda x: x[1])
    df['sugar'] = df['nutrition_parsed'].apply(lambda x: x[2])
    df['sodium'] = df['nutrition_parsed'].apply(lambda x: x[3])
    df['protein'] = df['nutrition_parsed'].apply(lambda x: x[4])
    df['sat_fat'] = df['nutrition_parsed'].apply(lambda x: x[5])
    df['carbs'] = df['nutrition_parsed'].apply(lambda x: x[6])

    # convert ingredients list → string
    df['ingredients'] = df['ingredients'].apply(
        lambda x: " ".join(ast.literal_eval(x)) if isinstance(x, str) else ""
    )

    df['cleaned_ingredients'] = df['ingredients'].apply(clean_ingredients)

    # Fix steps column
    if 'steps' in df.columns:
        df['steps'] = df['steps'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
    else:
        df['steps'] = [[] for _ in range(len(df))]

    return df

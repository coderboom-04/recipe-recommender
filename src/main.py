from preprocess import load_and_clean_data
from model import RecipeRecommender

df=load_and_clean_data("RAW_recipes.csv")
model=RecipeRecommender(df)

print(df.columns.tolist())

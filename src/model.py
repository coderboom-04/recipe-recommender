import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecipeRecommender:
    def __init__(self, dataframe):
        self.df = dataframe

        # TF–IDF Vectorizer on cleaned_ingredients
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["cleaned_ingredients"])

    def recommend_semantic(self, user_input, top_n=5):
        """Simple TF-IDF semantic similarity"""
        
        if not user_input.strip():
            return pd.DataFrame()  # return empty if user enters blank input

        # Convert user input into same TF-IDF space
        user_vec = self.vectorizer.transform([user_input])

        # Compute similarity
        sims = cosine_similarity(user_vec, self.tfidf_matrix).flatten()

        # Top matches
        top_idx = sims.argsort()[-top_n:][::-1]

        results = self.df.iloc[top_idx].copy()
        results["semantic_score"] = (sims[top_idx] * 100).round(2)

        return results

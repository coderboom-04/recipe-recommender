import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class RecipeRecommender:
    def __init__(self, dataframe):
        self.df = dataframe

        # Load a lightweight cloud-friendly model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Create embeddings for ingredients
        cleaned_list = self.df['cleaned_ingredients'].tolist()
        self.recipe_embeddings = self.model.encode(cleaned_list, convert_to_numpy=True)

        # TF-IDF optional (not used now)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["cleaned_ingredients"])

    def recommend_semantic(self, user_input, top_n=5):
        query_vec = self.model.encode([user_input], convert_to_numpy=True)

        sims = cosine_similarity(query_vec, self.recipe_embeddings).flatten()

        top_idx = sims.argsort()[-top_n:][::-1]

        results = self.df.iloc[top_idx].copy()
        results["semantic_score"] = (sims[top_idx] * 100).round(2)

        return results

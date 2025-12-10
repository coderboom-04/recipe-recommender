import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import TextEmbedding

class RecipeRecommender:
    def __init__(self, dataframe):
        self.df = dataframe

        # Fast lightweight embedding model (cloud friendly)
        self.embedding_model = TextEmbedding()

        cleaned_list = self.df["cleaned_ingredients"].tolist()
        self.recipe_embeddings = np.array(
            list(self.embedding_model.embed(cleaned_list))
        )

    def recommend_semantic(self, user_input, top_n=5):
        user_vec = np.array(
            list(self.embedding_model.embed([user_input]))
        ).reshape(1, -1)

        sims = cosine_similarity(user_vec, self.recipe_embeddings).flatten()

        top_idx = sims.argsort()[-top_n:][::-1]

        results = self.df.iloc[top_idx].copy()
        results["semantic_score"] = (sims[top_idx] * 100).round(2)

        return results

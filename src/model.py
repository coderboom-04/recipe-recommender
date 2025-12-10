import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")
@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None})
def embed_list(_nlp, text_list):
    return np.array([_nlp(text).vector for text in text_list])


class RecipeRecommender:
    def __init__(self, dataframe):
        self.df = dataframe

        # TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['cleaned_ingredients'])

        # spaCy model
        self.nlp = load_spacy_model()

        # Precompute embeddings
        cleaned_list = self.df['cleaned_ingredients'].tolist()
        self.recipe_embeddings = embed_list(self.nlp, cleaned_list)


    def recommend_semantic(self, user_input, top_n=5):
        query_vec = self.nlp(user_input).vector.reshape(1, -1)
        sims = cosine_similarity(query_vec, self.recipe_embeddings).flatten()

        top_idx = sims.argsort()[-top_n:][::-1]

        results = self.df.iloc[top_idx].copy()
        results["semantic_score"] = (sims[top_idx] * 100).round(2)
        return results

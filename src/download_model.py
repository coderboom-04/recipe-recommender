from sentence_transformers import SentenceTransformer

print("Downloading model... please wait...")
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
model.save("D:/recipe-recommender/models/miniLM-L3")
print("DONE! Model saved.")

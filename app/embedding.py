# embedding.py

import numpy as np
from config import config

# Load your embedding model (the actual model loading logic should be defined here)
def load_embedding_model():
    """
    Load the embedding model based on the configuration.
    """
    # Assuming the function load_embedding_model loads the model from the config
    # This would typically involve something like: SentenceTransformer or other embedding models.
    embedding_model = config.embedding_model
    return embedding_model

def compute_embeddings(text, model):
    """
    Generate the embedding for a given text using the embedding model.
    """
    return model.encode(text)

def compute_embeddings_for_dataset(tutorial_data, embedding_model):
    """
    Compute the embeddings for each document in the dataset and store them in memory.
    """
    vector_db = {}
    for example in tutorial_data:
        text = example['title'] + " " + example['document']
        embedding = embedding_model.encode(text)  # Generate the embedding for the document
        vector_db[embedding.tobytes()] = text  # Store the embedding and text in memory
    print("Embeddings computed and stored in memory.")
    return vector_db

# chat.py

from data_loader import load_tutorial_data
from embedding import load_embedding_model, compute_embeddings, compute_embeddings_for_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, set_seed

# Load the GPT-2 text generation pipeline
generator = pipeline('text-generation', model='gpt2')

# Optionally set a seed for reproducibility
set_seed(42)

# Load the dataset
tutorial_data = load_tutorial_data()

# Load the embedding model
embedding_model = load_embedding_model()

# Compute embeddings for all documents in memory and store them in a dictionary
vector_db = compute_embeddings_for_dataset(tutorial_data, embedding_model)

def chat_with_user(user_query):
    """
    Process the user's query, check the embeddings for relevant responses,
    and fall back to GPT-2 if no relevant content is found in the dataset.
    """
    # Compute the embedding for the user query
    query_embedding = compute_embeddings(user_query, embedding_model)

    # Check for relevant results in the vector database (in-memory)
    closest_matches = search_in_vector_db(query_embedding, vector_db)

    if closest_matches:
        return format_response(closest_matches, source="dataset")

    # If no match, fall back to GPT-2 text generation
    return generate_llm_response(user_query)

def search_in_vector_db(query_embedding, vector_db):
    """
    Search the in-memory vector database using cosine similarity and return the most relevant document(s).
    """
    results = []

    # Print once at the start
    print(f"Query embedding shape: {query_embedding.shape}")

    # Iterate through the embeddings in the vector DB (stored in memory)
    for stored_embedding, doc_text in vector_db.items():
        stored_embedding_array = np.frombuffer(stored_embedding, dtype=np.float32)

        # Compute cosine similarity
        similarity_score = cosine_similarity([query_embedding], [stored_embedding_array])[0][0]

        # Append results if similarity score exceeds a reasonable threshold
        if similarity_score > 0.75:
            results.append((similarity_score, doc_text))

    # Sort the results based on the similarity score
    results.sort(reverse=True, key=lambda x: x[0])

    return results[:3] if results else None

def generate_llm_response(user_query):
    """
    Generate a fallback response using GPT-2 if no relevant data is found in the dataset.
    """
    # Use GPT-2 to generate a response for the user query with explicit truncation and padding settings
    response = generator(
        user_query, 
        max_length=550, 
        num_return_sequences=1, 
        truncation=True,  # Enable truncation to handle long inputs
        pad_token_id=50256  # Set pad_token_id to the eos_token_id of GPT-2 (50256 is the default for GPT-2)
    )[0]['generated_text']

    return format_response(response, source="llm")

def format_response(response, source="dataset"):
    """
    Enhanced formatting of the response for better readability and user engagement.
    - If it's from the dataset, provides confidence and context.
    - If it's from the LLM (GPT-2), clearly indicates that it is AI-generated.
    """
    if isinstance(response, list):
        # Handle multiple matches from the dataset
        formatted_response = "<div class='response-header'>Here are the top results I found in the dataset:</div><ul>"
        for i, (similarity, doc_text) in enumerate(response):
            formatted_response += f"""
            <li>
                <strong>Result {i+1}</strong><br>
                <strong>Confidence</strong>: {similarity*100:.2f}%<br>
                <strong>Summary</strong>: {doc_text[:1500]}...<br>
            </li><br>
            """
        formatted_response += "</ul>"
        return formatted_response
    else:
        if source == "llm":
            return f"AI Tutor (GPT-2 generated): {response}\n"
        else:
            return f"AI Tutor (Dataset retrieved): {response}"

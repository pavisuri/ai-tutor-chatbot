# **AI Tutor Chatbot Project**

This project implements an AI Tutor chatbot using the **Hugging Face dataset** [Pavithrars/AI_Tutor](https://huggingface.co/datasets/Pavithrars/AI_Tutor), the **BAAI/bge-small-en-v1.5** embedding model for efficient document retrieval, and **GPT-2** for generating responses when no relevant documents are found in the dataset.

## **Features**

- **AI Tutorials Dataset**: Uses a comprehensive dataset from Hugging Face to provide accurate information on various AI-related topics.
- **Efficient Embedding Retrieval**: Generates embeddings using the **BAAI/bge-small-en-v1.5** model, storing them in a vector store for fast and accurate search.
- **Fallback to GPT-2**: If no match is found in the vector store, the system falls back to **GPT-2** to generate creative and context-aware responses.
- **Web Interface**: Implemented using **Flask**, allowing users to interact with the AI Tutor chatbot via a simple web form.

## **Requirements**

- Python 3.7 or higher
- Hugging Face `datasets` library for loading and managing the AI tutorials dataset
- Hugging Face `transformers` for GPT-2 text generation and the BAAI embedding model
- `Flask` for the web-based interface to interact with the AI Tutor
- `requests` library (used for possible future integrations or external API calls)
- `numpy` for numerical computations, particularly for working with embeddings
- `scikit-learn` for cosine similarity calculations

### **Required Python Packages**

```txt
Flask==2.0.3
datasets==2.6.1
transformers==4.21.0
sentence-transformers==2.2.2
numpy==1.21.2
scikit-learn==0.24.2

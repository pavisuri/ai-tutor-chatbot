class ChatConfig:
    def __init__(self, embedding_model, system_message, max_tokens, temperature, llm_provider, api_key):
        self.embedding_model = embedding_model
        self.system_message = system_message
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm_provider = llm_provider
        self.api_key = api_key

# Configuration object
config = ChatConfig(
    embedding_model="BAAI/bge-small-en-v1.5",  # <-- embedding model
    system_message="You are a helpful assistant.",
    max_tokens=256,
    temperature=0.7,
    llm_provider="ai_studio",
    api_key="API_KEY"
)

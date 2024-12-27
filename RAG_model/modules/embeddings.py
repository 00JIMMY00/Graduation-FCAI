import cohere

def generate_embeddings(text):
    client = cohere.Client(COHERE_API_KEY)
    response = client.embed(texts=[text])
    return response.embeddings[0]

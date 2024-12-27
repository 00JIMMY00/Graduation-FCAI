from modules.file_handlers import download_file, extract_text_from_pdf
from modules.vector_store import insert_vector
from modules.embeddings import generate_embeddings
from modules.chatbot import get_chat_response

def main():
    # Example workflow
    file_id = "example_id"
    file_path = download_file(file_id)
    text = extract_text_from_pdf(file_path)
    embedding = generate_embeddings(text)
    insert_vector({"text": text, "embedding": embedding})
    context = "retrieved context here"
    user_query = "What is this document about?"
    response = get_chat_response(user_query, context)
    print(response)

if __name__ == "__main__":
    main()

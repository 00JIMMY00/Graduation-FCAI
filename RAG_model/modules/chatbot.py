import openai

def get_chat_response(user_input, context):
    prompt = f"You are a helpful assistant. Based on the context: {context}, answer the user's query: {user_input}"
    response = openai.Completion.create(engine="gpt-4", prompt=prompt)
    return response.choices[0].text

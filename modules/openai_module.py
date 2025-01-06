import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (including OPENAI_API_KEY)
load_dotenv()

# Create an instance of the OpenAI Class using the API key
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_text_basic(
    prompt: str, 
    model: str, 
    temperature: float = 0.0, 
    top_p: float = 1.0,
) -> str:
    """
    Generates text using the chat.completions.create endpoint, 
    with optional temperature and top_p parameters.
    
    :param prompt: The user prompt.
    :param model: The model ID (e.g., "gpt-4o" or "gpt-4o-mini").
    :param temperature: Sampling temperature (0..2). Higher = more creative, lower = more strict.
    :param top_p: Nucleus sampling (0..1). 
    :return: The assistant's message content as a string.
    """
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p
    )
    return response.choices[0].message.content

def generate_text_with_conversation(messages, model="gpt-3.5-turbo") -> str:
    """
    Uses the same chat.completions.create endpoint but allows passing
    a list of 'messages' for multi-turn conversations.
    
    :param messages: List of message dicts, e.g. [{"role": "user", "content": "Hi"}]
    :param model: The model ID (default: "gpt-3.5-turbo").
    :return: The assistant's message content as a string.
    """
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

import os
from dotenv import load_dotenv
from openai import OpenAI
import json

# Load environment variables (including OPENAI_API_KEY)
load_dotenv()

# Create an instance of the OpenAI Class using the API key
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_text_basic(
    prompt: str, 
    model: str, 
    temperature: float = 1.0, 
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


def generate_text_with_conversation(messages, model="gpt-4o-mini") -> str:
    """
    Uses the same chat.completions.create endpoint but allows passing
    a list of 'messages' for multi-turn conversations.
    
    :param messages: List of message dicts, e.g. [{"role": "user", "content": "Hi"}]
    :param model: The model ID (default: "gpt-4o-mini").
    :return: The assistant's message content as a string.
    """
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content



def generate_text_with_function_call(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    debug: bool = False
) -> list:
    """
    Generates text using OpenAI's function-calling feature to ensure structured JSON output.

    Parameters
    ----------
    prompt : str
        The user prompt.
    model : str
        The model ID (e.g., "gpt-4" or "gpt-4-turbo").
    temperature : float, optional
        Sampling temperature (0..2). Higher = more creative, lower = more strict.
    top_p : float, optional
        Nucleus sampling (0..1).
    debug : bool, optional
        If True, include the "Reason" field in the response.

    Returns
    -------
    list
        A list of dictionaries representing the structured JSON response.
    """
    # Define the base schema
    function_schema = {
        "name": "process_row_decisions",
        "description": "Process row decisions for filtering based on conceptual reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "RowIndex": {"type": "integer", "description": "Index of the row being evaluated."},
                            "Decision": {
                                "type": "string",
                                "description": "Decision for the row: KEEP or EXCLUDE.",
                                "enum": ["KEEP", "EXCLUDE"]
                            },
                        },
                        "required": ["RowIndex", "Decision"]
                    }
                }
            },
            "required": ["rows"]
        }
    }

    # Add the "Reason" field if debug is enabled
    if debug:
        function_schema["parameters"]["properties"]["rows"]["items"]["properties"]["Reason"] = {
            "type": "string",
            "description": "Reason for the decision, included if debug=True.",
            "nullable": True
        }

    # Make the API call with the function schema
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        functions=[function_schema],
        function_call={"name": "process_row_decisions"},
        temperature=temperature,
        top_p=top_p
    )

    # Extract the function call arguments (structured JSON response)
    function_args = response.choices[0].message.function_call.arguments

    # Parse the structured JSON response
    try:
        structured_data = json.loads(function_args)["rows"]
        return structured_data
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to decode function response: {e}")


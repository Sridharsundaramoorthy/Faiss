import openai
from typing import List

def get_embedding_for_text(text: str, api_key: str) -> List[float]:
    """
    Generate an embedding for the given text using OpenAI's API
    
    Args:
        text: The text to generate an embedding for
        api_key: OpenAI API key
        
    Returns:
        The embedding vector
    """
    # Set the API key
    openai.api_key = api_key
    
    # Clean and prepare the text
    text = text.replace("\n", " ")
    
    try:
        # Get embedding from OpenAI
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        
        # Extract the embedding vector
        embedding = response['data'][0]['embedding']
        return embedding
    
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise
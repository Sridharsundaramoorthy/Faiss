import openai
from typing import List, Dict, Any
import time

def generate_answer_from_results(
    query: str, 
    results: List[Dict[str, Any]], 
    api_key: str,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Generate a meaningful answer from search results using OpenAI's LLM
    
    Args:
        query: The user's original query
        results: The search results from vector search
        api_key: OpenAI API key
        model: The OpenAI model to use
        
    Returns:
        A consolidated answer based on the search results
    """
    # Set the API key
    openai.api_key = api_key
    
    # Prepare context from search results
    context = ""
    for i, result in enumerate(results):
        context += f"\nDocument {i+1} (Similarity: {result['similarity_score']:.2f}):\n{result['text']}\n"
    
    print(f"[INFO] Generating answer using model: {model}")
    print(f"[INFO] Context length: {len(context)} characters from {len(results)} documents")
    
    # Prepare the prompt
    system_message = """You are a helpful assistant that provides accurate answers based on the given context.
    If the context doesn't contain relevant information to answer the question, say "I don't have enough information to answer this question."
    Always cite which document(s) you used for your answer."""

    # Add retry logic
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Get completion from OpenAI
            start_time = time.time()
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
                ],
                temperature=0.3,  # Lower temperature for more focused answers
                max_tokens=1000
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract the answer
            answer = response['choices'][0]['message']['content']
            print(f"[SUCCESS] Generated answer in {elapsed_time:.2f}s")
            print(f"[DEBUG] Answer preview: {answer[:100]}...")
            
            return answer
        
        except Exception as e:
            print(f"[ERROR] Attempt {attempt+1}/{max_retries}: Error generating answer: {e}")
            if attempt < max_retries - 1:
                print(f"[INFO] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"[ERROR] Failed to generate answer after {max_retries} attempts")
                return f"Error generating answer: {str(e)}"

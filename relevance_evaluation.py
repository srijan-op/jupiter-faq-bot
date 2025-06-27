import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv
import json  # <--- FIX 1: Import the json module

# Import your bot classes
from faq_bot import JupiterFAQBot
from llm_only_bot import LLMOnlyFAQBot

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.WARNING)

# Load the sentence transformer model once
print("[SETUP] Loading embedding model...")
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[SETUP] Loading LLM evaluator...")
try:
    # --- FIX 2: Initialize the LLM with JSON mode ---
    evaluator_llm = JupiterFAQBot().llm.bind(
        response_format={"type": "json_object"},
    )
except Exception as e:
    print(f"\n[FATAL ERROR] Could not initialize bots. Is GROQ_API_KEY set and vectorstore present? Details: {e}")
    exit()

# --- Core Functions ---

def get_embedding(text: str) -> np.ndarray:
    """Generates a sentence embedding for a given text."""
    return EMBEDDING_MODEL.encode(text, convert_to_tensor=False)

def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculates the cosine similarity between two embeddings."""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def evaluate_answer_relevance(query: str, response: str) -> dict:
    """
    Uses an LLM to evaluate an answer's relevance and semantic quality.
    """
    # The prompt is updated slightly to be more explicit for the model
    prompt = f"""
    You are an expert AI evaluator. Assess the following AI-generated answer based on the user's query.
    Provide your response as a JSON object with three keys: "relevance_score", "completeness_score", and "clarity_score", along with a "justification" key.
    - "relevance_score": Does the answer directly address the user's query? (1-5)
    - "completeness_score": Is the answer thorough and complete? (1-5)
    - "clarity_score": Is the answer easy to understand? (1-5)
    - "justification": A one-sentence explanation for your scores.

    User Query: "{query}"
    AI's Answer: "{response}"
    """
    try:
        evaluation = evaluator_llm.invoke(prompt).content
        # The output should now be a clean JSON string
        return json.loads(evaluation)
    except Exception as e:
        print(f"Error parsing LLM evaluation: {e}")
        return {
            "relevance_score": 0, "completeness_score": 0, "clarity_score": 0,
            "justification": "Failed to get evaluation from LLM."
        }

# --- Main Execution ---

if __name__ == "__main__":
    # Initialize the bot you want to test
    rag_bot = JupiterFAQBot()

    # --- Test Cases ---
    test_queries = [
        "How do I earn Jewels on International payments?", # A direct, specific question
        "Tell me about loans", # A general, vague question
        "Does Jupiter sell pizza?" # An out-of-scope, irrelevant question
    ]

    for query in test_queries:
        print("\n" + "="*50)
        print(f"EVALUATING QUERY: \"{query}\"")
        print("="*50)

        # 1. Get the bot's response
        response_data = rag_bot.get_response(query)
        response_text = response_data['response']
        print(f"\n[Bot Response]:\n{response_text}\n")

        # 2. Calculate mathematical semantic similarity
        query_embedding = get_embedding(query)
        response_embedding = get_embedding(response_text)
        cosine_score = calculate_cosine_similarity(query_embedding, response_embedding)

        # 3. Get LLM-based relevance evaluation
        llm_evaluation = evaluate_answer_relevance(query, response_text)

        # 4. Print the final report
        print("--- Evaluation Report ---")
        print(f"Mathematical Semantic Similarity (Cosine): {cosine_score:.4f}")
        print(f"LLM Judgement - Relevance Score:    {llm_evaluation.get('relevance_score', 'N/A')}/5")
        print(f"LLM Judgement - Completeness Score: {llm_evaluation.get('completeness_score', 'N/A')}/5")
        print(f"LLM Judgement - Clarity Score:      {llm_evaluation.get('clarity_score', 'N/A')}/5")
        print(f"LLM Justification: {llm_evaluation.get('justification', 'N/A')}")
        print("--- End of Report ---\n")
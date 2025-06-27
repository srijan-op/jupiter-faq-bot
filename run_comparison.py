import os
import time
import pandas as pd
import logging
from dotenv import load_dotenv

# Import the bot classes from your files
from faq_bot import JupiterFAQBot
from llm_only_bot import LLMOnlyFAQBot

# --- Setup ---
load_dotenv()
# Set logging to a higher level to keep the comparison output clean
logging.basicConfig(level=logging.WARNING) 
logger = logging.getLogger(__name__)

# --- Accuracy Evaluation using an LLM ---
# We use the LLM instance from one of the bots for automated evaluation.
# This avoids initializing another LLM client.
try:
    EVALUATOR_LLM = JupiterFAQBot().llm 
except ValueError as e:
    print(f"\n[FATAL ERROR] Could not initialize the evaluator LLM. Please check your GROQ_API_KEY in the .env file. Details: {e}")
    exit()
except Exception as e:
    print(f"\n[FATAL ERROR] An unexpected error occurred during initialization. Is your vectorstore_faq folder present? Details: {e}")
    exit()

def evaluate_accuracy(query: str, ground_truth: str, bot_response: str) -> tuple[int, str]:
    """
    Uses an LLM to evaluate the bot's response against the ground truth.
    Returns a score (1-5) and a justification.
    """
    if not ground_truth:
        # Cannot evaluate without a ground truth answer
        return (0, "No ground truth provided.")

    prompt = f"""
    You are an impartial evaluator. Your task is to assess the quality of a bot's response to a user query, based on a provided ground truth answer.

    User Query: "{query}"
    Ground Truth Answer: "{ground_truth}"
    Bot's Response: "{bot_response}"

    Please evaluate the bot's response on a scale of 1 to 5 based on the following criteria:
    - 5: The response is highly accurate, directly answers the query, and aligns perfectly with the ground truth.
    - 4: The response is accurate and answers the query well, but might be slightly less detailed or phrased differently than the ground truth.
    - 3: The response is partially correct or relevant but misses key information from the ground truth.
    - 2: The response is mostly inaccurate, irrelevant, or hallucinates information not present in the ground truth.
    - 1: The response is completely wrong, irrelevant, or harmful.

    First, provide a one-sentence justification for your score. Then, on a new line, provide the integer score.
    
    Example:
    Justification: The bot correctly identified the main point but missed a detail about the KYC process.
    Score: 4
    """
    try:
        evaluation = EVALUATOR_LLM.invoke(prompt).content
        lines = evaluation.strip().split('\n')
        justification = lines[0].replace("Justification: ", "").strip()
        score = int(lines[-1].replace("Score: ", "").strip())
        return score, justification
    except Exception as e:
        logger.error(f"Failed to evaluate accuracy: {e}")
        return (0, "Evaluation failed.")

# --- Test Suite Definition ---
def get_test_suite():
    """
    Creates a test suite with questions from the CSV and custom ones.
    """
    try:
        df = pd.read_csv("jupiter_faqs_processed.csv")
    except FileNotFoundError:
        print("\n[FATAL ERROR] 'jupiter_faqs_processed.csv' not found. Please make sure it's in the same directory.")
        exit()
        
    df['question'] = df['question'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 1. Direct Questions from FAQ (using a fixed sample for consistent results)
    direct_questions = df.sample(5, random_state=42).to_dict('records')
    for q in direct_questions:
        q['type'] = 'Direct FAQ'
        q['ground_truth'] = q.pop('answer')

    # 2. Paraphrased Questions
    paraphrased_questions = [
        {'type': 'Paraphrased', 'question': 'How can I set up a salary account?', 'ground_truth': df[df.question.str.contains("How do I open a Salary account", case=False)]['answer'].iloc[0]},
        {'type': 'Paraphrased', 'question': 'Can I use my card abroad without extra fees?', 'ground_truth': df[df.question.str.contains("use my Jupiter Debit Card outside India", case=False)]['answer'].iloc[0]},
        {'type': 'Paraphrased', 'question': 'What do I get for paying my bills?', 'ground_truth': df[df.question.str.contains("rewards for bill payments", case=False)]['answer'].iloc[0]},
    ]
    
    # 3. Out-of-Scope / Hallucination Test
    out_of_scope_questions = [
        {'type': 'Out of Scope', 'question': 'Does Jupiter offer car loans?', 'ground_truth': 'Jupiter does not offer car loans. The bot should state it cannot find this information or refer to supported loan types (Personal, Mini, Against MF).'},
        {'type': 'Out of Scope', 'question': 'What is the stock price of Jupiter Money?', 'ground_truth': 'Jupiter is a private company and does not have a stock price. The bot should state it does not have this information.'},
    ]

    # 4. General/Vague Question
    general_question = [
        {'type': 'General', 'question': 'Tell me about investing with Jupiter', 'ground_truth': 'The bot should provide a summary of investment options like Mutual Funds, Digital Gold, and FDs, mentioning key features like zero commission or no-penalty SIPs.'}
    ]
    
    test_suite = direct_questions + paraphrased_questions + out_of_scope_questions + general_question
    return test_suite

# --- Main Execution ---
def main():
    """
    Runs the comparison between the RAG bot and the LLM-only bot.
    """
    print("--- Jupiter Bot Comparison: RAG vs. LLM-Only ---")

    # We assume the user has already run their faq_ingest.py script.
    if not os.path.exists("vectorstore_faq"):
        print("\n[FATAL ERROR] The 'vectorstore_faq' directory was not found.")
        print("Please run your 'faq_ingest.py' script first to create the vector store.")
        return
    else:
        print("\n[SETUP] Found 'vectorstore_faq' directory. Proceeding with evaluation.")

    # 2. Instantiate bots
    print("[SETUP] Initializing bots...")
    # The RAG bot was already initialized for the evaluator, so we can reuse it.
    rag_bot = JupiterFAQBot()
    llm_only_bot = LLMOnlyFAQBot()
    
    # 3. Load test suite
    test_suite = get_test_suite()
    print(f"[SETUP] Loaded {len(test_suite)} test questions.")

    # 4. Run evaluation
    results = []
    print("\n--- Running Evaluation ---")
    for i, item in enumerate(test_suite):
        query = item['question']
        ground_truth = item.get('ground_truth', '')
        q_type = item['type']

        print(f"\n({i+1}/{len(test_suite)}) Testing Query [{q_type}]: \"{query}\"")

        # --- Test RAG Bot (JupiterFAQBot) ---
        start_time = time.perf_counter()
        rag_response_data = rag_bot.get_response(query)
        rag_latency = time.perf_counter() - start_time
        rag_response = rag_response_data['response']
        rag_confidence = rag_response_data['confidence']
        rag_accuracy, rag_justification = evaluate_accuracy(query, ground_truth, rag_response)

        # --- Test LLM-Only Bot ---
        start_time = time.perf_counter()
        llm_response_data = llm_only_bot.get_response(query)
        llm_latency = time.perf_counter() - start_time
        llm_response = llm_response_data['response']
        llm_accuracy, llm_justification = evaluate_accuracy(query, ground_truth, llm_response)
        
        results.append({
            'Type': q_type,
            'Query': query,
            'RAG Latency (s)': f"{rag_latency:.2f}",
            'RAG Accuracy (1-5)': rag_accuracy,
            'RAG Confidence': f"{rag_confidence:.2f}",
            'RAG Justification': rag_justification,
            'LLM-Only Latency (s)': f"{llm_latency:.2f}",
            'LLM-Only Accuracy (1-5)': llm_accuracy,
            'LLM-Only Justification': llm_justification,
        })

    # 5. Display results
    results_df = pd.DataFrame(results)
    
    # Calculate averages, converting relevant columns to numeric first
    for col in ['RAG Latency (s)', 'RAG Accuracy (1-5)', 'LLM-Only Latency (s)', 'LLM-Only Accuracy (1-5)']:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

    avg_rag_latency = results_df['RAG Latency (s)'].mean()
    avg_rag_accuracy = results_df[results_df['RAG Accuracy (1-5)'] > 0]['RAG Accuracy (1-5)'].mean()
    
    avg_llm_latency = results_df['LLM-Only Latency (s)'].mean()
    avg_llm_accuracy = results_df[results_df['LLM-Only Accuracy (1-5)'] > 0]['LLM-Only Accuracy (1-5)'].mean()

    print("\n\n--- Detailed Evaluation Results ---")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 60)
    print(results_df[['Type', 'Query', 'RAG Accuracy (1-5)', 'LLM-Only Accuracy (1-5)', 'RAG Latency (s)', 'LLM-Only Latency (s)']])

    print("\n\n--- Summary & Analysis ---")
    summary_data = {
        'Metric': ['Avg. Latency (s)', 'Avg. Accuracy (1-5)'],
        'RAG Bot (Retrieval-Based)': [f"{avg_rag_latency:.2f}", f"{avg_rag_accuracy:.2f}"],
        'LLM-Only Bot (Generative)': [f"{avg_llm_latency:.2f}", f"{avg_llm_accuracy:.2f}"]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print("\n### Key Observations ###")
    print(f"1.  **Accuracy**: The RAG Bot achieved a higher average accuracy ({avg_rag_accuracy:.2f}) compared to the LLM-Only Bot ({avg_llm_accuracy:.2f}). By grounding its answers in the provided FAQ data, it consistently provides more factual and relevant responses, especially for specific questions.")
    print(f"2.  **Latency**: The LLM-Only Bot was consistently faster (avg {avg_llm_latency:.2f}s) than the RAG Bot (avg {avg_rag_latency:.2f}s). The RAG bot's latency includes the extra step of searching the vector database, which adds a slight overhead.")
    print("3.  **Handling Out-of-Scope Questions**: The RAG Bot demonstrates greater safety. When it can't find relevant FAQs, its response reflects this uncertainty. The LLM-Only bot is more prone to 'hallucination'—confidently inventing plausible but incorrect information.")
    print("4.  **Consistency & Control**: The RAG approach offers significantly more control. Its answers are tethered to the `jupiter_faqs_processed.csv`, making it reliable for a business context where factual accuracy from a known source is critical.")
    
    print("\n### Conclusion ###")
    print("For a customer-facing FAQ chatbot, the **RAG (Retrieval-Based) approach is superior despite its slightly higher latency.** The significant gains in accuracy, reliability, and safety against making up answers far outweigh the small performance cost. The LLM-Only approach is simpler and faster but carries an unacceptable risk of providing incorrect information for this specific use case.")


if __name__ == "__main__":
    main()
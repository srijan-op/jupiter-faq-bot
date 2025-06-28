# Jupiter FAQ Bot 🤖: A RAG based Q&A Bot 

Deployed link : https://jupiter-faq-bot-ons3kklxc5bq63goavcwot.streamlit.app/

This project delivers a comprehensive solution for building and evaluating a conversational FAQ bot based on the Jupiter Help Centre, fulfilling all core and bonus objectives of the assignment.

The repository contains two distinct chatbot architectures:
1.  **Retrieval-Augmented Generation (RAG):** A sophisticated bot that retrieves factual context from a knowledge base before generating an answer.
2.  **LLM-Only:** A baseline bot that relies solely on the general knowledge of a large language model.

A Streamlit application provides a user-friendly interface to interact with both bots, and a suite of evaluation scripts offers a deep, data-driven comparison of their performance.


---

## 🏗️ Architecture & Workflow

The key difference between the two bots is how they process a user's query.

### Retrieval-Augmented Generation (RAG) Workflow

The RAG bot takes a multi-step approach to ensure answers are factual and context-aware. It *retrieves* information before it *generates* an answer.

```
+-----------+         +-----------------+         +---------------------+
| User      |   1.    | RAG Bot         |   2.    | Vector Store        |
|-----------|  Query  |-----------------| Search  | (ChromaDB)          |
| Asks      |-------->| Embeds Query    |-------->|---------------------|
| Question  |         | & Searches      |         | Finds Similar       |
+-----------+         +-----------------+         | FAQ Questions       |
      ^                       |                   +---------------------+
      |                       | 3. Retrieved Context
      |                       V
      |             +-----------------+
      |             | Prompt          |
      |             |-----------------|
      |             | Original Query  |
      |             | + Relevant FAQs |
      |             +-----------------+
      |                       | 4. Rich Prompt
      |                       V
      |             +-----------------+
      |       5.    | LLM (Groq)      |
      |   Grounded  |-----------------|
      |    Answer   | Generates Final |
      |<----------- | Answer          |
      |             +-----------------+
      |
+-----------+
| User      |
| Receives  |
| Answer    |
+-----------+
```

### LLM-Only Workflow

The LLM-Only bot has a much simpler but less reliable workflow. It relies entirely on the LLM's pre-existing knowledge, which can be out-of-date or incorrect for specific topics.

```
+-----------+      +-----------------+      +--------------+
| User      |  1.  | LLM-Only Bot    |  2.  | LLM (Groq)   |
|-----------| Query|-----------------|Prompt|--------------|
| Asks      |----->| Wraps Query in  |----->| Generates    |
| Question  |      | Simple Prompt   |      | Answer from  |
+-----------+      +-----------------+      | General      |
      ^                                     | Knowledge    |
      |                                     +--------------+
      |                  3. Generic Answer
      |<------------------------------------------
      |
+-----------+
| User      |
| Receives  |
| Answer    |
+-----------+
```

---

## 📂 Project Structure

```
Jupiter-faq-bot/
├── vectorstore_faq/        # Populated by the ingest script
├── .env                    # Local environment variables (GITIGNORED)
├── .gitignore
├── app.py                  # The Streamlit web interface
├── faq_bot.py              # RAG Bot implementation
├── llm_only_bot.py         # LLM-Only Bot implementation
├── faq_ingest.py           # Script to create the vector store
├── run_comparison.py       # Main script for accuracy/latency test
├── evaluate_relevance.py   # Script for deep semantic analysis
├── jupiter_faqs_processed.csv
├── requirements.txt
└── README.md               # This file
```

---

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/srijan-op/jupiter-faq-bot
    cd Jupiter-faq-bot
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Create the environment file:**
    Create a file named `.env` in the root directory and add your Groq API key:
    ```
    GROQ_API_KEY="your-groq-api-key-here"
    ```

---

## 🚀 How to Run the Project

Follow these steps in order.

### 1️⃣ Populate the Knowledge Base (One-Time Setup)

Run the data ingestion script to create the `vectorstore_faq` directory from the CSV data.

```bash
python faq_ingest.py
```

### 2️⃣ Launch the Interactive Web App

This is the primary way to demonstrate the solution. It allows you to switch between the RAG and LLM-Only bots in real-time.

```bash
streamlit run app.py
```

### 3️⃣ Run the Evaluation Scripts (Optional)

These scripts provide the data-driven analysis of the bots' performance.

**A. Main Comparison (Accuracy vs. Latency)**
```bash
python run_comparison.py
```

**B. Deep Semantic Relevance Evaluation**
```bash
python evaluate_relevance.py
```

---
## 📈 Analysis of Results

The evaluation scripts provide concrete data highlighting the strengths and weaknesses of each architecture.

### A. Accuracy vs. Latency Comparison

The `run_comparison.py` script provides a high-level overview of performance across a varied test suite.

**Results from `run_comparison.py`:**
```
--- Summary & Analysis ---
                 Metric RAG Bot (Retrieval-Based) LLM-Only Bot (Generative)
   Avg. Latency (s)                       1.21                      0.48
Avg. Accuracy (1-5)                       4.75                      3.60
```
**Interpretation:** The data shows a clear trade-off. The RAG Bot is more accurate (4.75/5) but slower (1.21s), while the LLM-Only bot is faster (0.48s) but significantly less accurate (3.60/5). For a customer-facing application, the gain in accuracy and trustworthiness far outweighs the sub-second difference in latency.

### B. Semantic Relevance & Quality

The `evaluate_relevance.py` script performs a deeper analysis of the RAG bot's response quality for a specific query.

**Results for query: "How do I earn Jewels on International payments?"**
```
--- Evaluation Report ---
Mathematical Semantic Similarity (Cosine): 0.8874
LLM Judgement - Relevance Score:    5/5
LLM Judgement - Completeness Score: 5/5
LLM Judgement - Clarity Score:      5/5
LLM Justification: The answer is a direct, complete, and clear response to the user's specific question.
```
**Interpretation:** The high cosine similarity (0.8874) confirms a strong semantic link between the question and the answer. The perfect scores from the LLM evaluator for relevance, completeness, and clarity demonstrate that the RAG bot produces high-quality, helpful, and contextually appropriate responses.

---

## 🏆 Conclusion

The comprehensive evaluation clearly demonstrates that the **Retrieval-Augmented Generation (RAG) architecture is superior** for this business use case. While the LLM-Only bot is marginally faster, the RAG bot provides significantly more accurate, relevant, and safe responses by grounding its answers in a verified knowledge base. This makes it the ideal choice for a customer-facing application where trust and factual correctness are paramount.

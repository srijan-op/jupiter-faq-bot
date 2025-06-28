# faq_bot.py

import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import Counter

from langchain_groq import ChatGroq
import chromadb
from chromadb.utils import embedding_functions

from langdetect import detect
from deep_translator import GoogleTranslator

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JupiterFAQBot:
    def __init__(self, chroma_path="vectorstore_faq", model_name="llama3-70b-8192"):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Missing GROQ_API_KEY in environment.")

        self.llm = ChatGroq(
            temperature=0.7,
            groq_api_key=self.groq_api_key,
            model_name=model_name
        )

        embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(name="faq_collection", embedding_function=embedding_model)

        self.confidence_threshold = 0.6
        self.conversation_history = []

    def _translate_to_english(self, text: str) -> str:
        try:
            lang = detect(text)
            if lang != "en":
                return GoogleTranslator(source='auto', target='en').translate(text)
            return text
        except:
            return text

    def _translate_to_original_language(self, text: str, original_lang: str) -> str:
        try:
            if original_lang != "en":
                return GoogleTranslator(source='en', target=original_lang).translate(text)
            return text
        except:
            return text

    def search_faqs(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        faqs = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            faq = {
                "question": doc,
                "answer": meta.get("answer", ""),
                "category": meta.get("category", "General"),
                "alternative_questions": meta.get("alternative_questions", [])
            }
            similarity = 1 - dist
            if similarity >= self.confidence_threshold:
                faqs.append((faq, similarity))
        return faqs

    def _get_similar_questions_from_chroma(self, query: str, exclude: str, top_k: int = 5) -> List[str]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k + 1,
                include=["documents"]
            )
            suggestions = []
            for doc in results["documents"][0]:
                if doc.strip().lower() != exclude.strip().lower():
                    suggestions.append(doc)
                if len(suggestions) >= top_k:
                    break
            return suggestions
        except Exception as e:
            logger.error(f"Error in _get_similar_questions_from_chroma: {e}")
            return []

    def generate_response_with_llm(self, query: str, relevant_faqs: List[Tuple[Dict, float]]) -> str:
        try:
            context = self._build_context(relevant_faqs)

            system_prompt = """You are a helpful customer service assistant for Jupiter, a digital banking app.
Your job is to provide friendly, conversational, and accurate answers based on the FAQ context provided.
If you don’t have enough information, respond politely and suggest contacting support."""

            user_prompt = f"""User Question: {query}
Relevant FAQ Info:
{context}

Respond in a simple, helpful, and human-like tone."""

            response = self.llm.invoke([
                ("system", system_prompt),
                ("user", user_prompt)
            ])
            return response.content.strip()

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._generate_simple_response(relevant_faqs)

    def get_response(self, query: str) -> Dict:
        logger.info(f"User Query: {query}")

        try:
            original_lang = detect(query)
        except:
            original_lang = 'en' # Default to English if detection fails
            
        translated_query = self._translate_to_english(query)

        if not translated_query.strip():
            return {
                "response": "Please ask a valid question about Jupiter's banking services.",
                "confidence": 0.0,
                "source_faqs": [],
                "suggestions": self._get_popular_questions()
            }

        relevant_faqs = self.search_faqs(translated_query)
        
        # --- FIX FOR RAG BOT ---
        # If no relevant FAQs are found, the query is off-topic. Return a standard, non-translated response.
        if not relevant_faqs:
            response_text = "I couldn't find anything specific. Can you please rephrase or find some relevant doubts from the category section? Please contact Jupiter support for more assistance."
            return {
                "response": response_text,
                "confidence": 0.0,
                "source_faqs": [],
                "suggestions": self._get_popular_questions()
            }
        
        # --- NORMAL FLOW ---
        # This part only runs if we found relevant FAQs.
        response_text = self.generate_response_with_llm(translated_query, relevant_faqs)
        response_text = self._translate_to_original_language(response_text, original_lang)

        confidence = relevant_faqs[0][1] if relevant_faqs else 0.0
        top_faq = relevant_faqs[0][0] if relevant_faqs else {}
        main_question = top_faq.get("question", "")

        suggestions = list(set(
            self._get_similar_questions_from_chroma(translated_query, exclude=main_question) +
            self._get_behavior_based_suggestions()
        ))[:3]

        self.conversation_history.append({
            "query": query,
            "response": response_text,
            "confidence": confidence,
            "timestamp": self._get_timestamp()
        })

        return {
            "response": response_text,
            "confidence": confidence,
            "source_faqs": [faq for faq, _ in relevant_faqs],
            "suggestions": suggestions
        }

    def _get_behavior_based_suggestions(self, recent_n: int = 50) -> List[str]:
        queries = [entry["query"] for entry in self.conversation_history[-recent_n:]]
        query_counts = Counter(queries)
        suggestions = [q for q, _ in query_counts.most_common(5)]
        return suggestions[:3]

    def _build_context(self, relevant_faqs: List[Tuple[Dict, float]]) -> str:
        return "\n".join(
            f"FAQ {i+1} (Score: {score:.2f}):\nQ: {faq['question']}\nA: {faq['answer']}\nCategory: {faq['category']}"
            for i, (faq, score) in enumerate(relevant_faqs)
        )

    def _generate_simple_response(self, relevant_faqs: List[Tuple[Dict, float]]) -> str:
        if not relevant_faqs:
            return "I'm not sure how to help with that. Please contact Jupiter support for more assistance."
        faq, score = relevant_faqs[0]
        if score < 0.7:
            return f"I'm not entirely sure, but this might help:\n\n{faq['answer']}"
        return f"Here’s what I found:\n\n{faq['answer']}"

    def _get_timestamp(self) -> str:
        return datetime.now().isoformat()

    def _get_popular_questions(self) -> List[str]:
        return [
            "How do I activate my debit card?",
            "What is the KYC process?",
            "How do I check Jupiter rewards?"
        ]

    def get_conversation_history(self) -> List[Dict]:
        return self.conversation_history

    def clear_history(self):
        self.conversation_history = []
        logger.info("Conversation history cleared.")

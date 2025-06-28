import os
import logging
from dotenv import load_dotenv
from datetime import datetime
from langdetect import detect
from deep_translator import GoogleTranslator
from langchain_groq import ChatGroq

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMOnlyFAQBot:
    def __init__(self, model_name="llama3-70b-8192"):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Missing GROQ_API_KEY in environment.")

        self.llm = ChatGroq(
            temperature=0.7,
            groq_api_key=self.groq_api_key,
            model_name=model_name
        )
        # A separate, zero-temperature model for classification
        self.classifier_llm = ChatGroq(
            temperature=0.0,
            groq_api_key=self.groq_api_key,
            model_name="llama3-8b-8192" # Use a smaller, faster model for this simple task
        )
        self.conversation_history = []

    def _translate_to_english(self, text: str) -> str:
        try:
            lang = detect(text)
            if lang != "en":
                return GoogleTranslator(source='auto', target='en').translate(text)
            return text
        except Exception as e:
            logger.warning(f"Translation to English failed: {e}")
            return text

    def _translate_to_original_language(self, text: str, original_lang: str) -> str:
        try:
            if original_lang != "en":
                return GoogleTranslator(source='en', target=original_lang).translate(text)
            return text
        except Exception as e:
            logger.warning(f"Translation back to original language failed: {e}")
            return text
            
    # --- NEW: Robust classification method ---
    def _is_query_on_topic(self, query: str) -> bool:
        """Uses a separate LLM call to classify the query's topic."""
        try:
            system_prompt = """You are a topic classifier. Your only job is to determine if the user's query is related to banking, finance, payments, accounts, cards, or the Jupiter app.
Answer with a single word: 'yes' or 'no'."""
            
            response = self.classifier_llm.invoke([
                ("system", system_prompt),
                ("user", query)
            ])
            
            answer = response.content.strip().lower()
            logger.info(f"Classifier result: '{answer}' for query: '{query}'")
            return "yes" in answer

        except Exception as e:
            logger.error(f"Classifier LLM error: {e}")
            # If classification fails, assume the query is on-topic to avoid blocking a valid question.
            return True

    def get_response(self, query: str) -> dict:
        logger.info(f"[LLM-Only] User Query: {query}")

        try:
            original_lang = detect(query)
        except:
            original_lang = 'en' # Default to English if detection fails

        translated_query = self._translate_to_english(query)

        if not translated_query.strip():
            return {
                "response": "Please ask a valid question about Jupiter's banking services.",
                "source_faqs": [],
                "suggestions": []
            }
        
        # --- ROBUST FIX: TWO-STEP PROCESS ---
        # 1. Classify the query first
        if not self._is_query_on_topic(translated_query):
            # If off-topic, return a standard, non-translated English response.
            response_text = "I'm sorry, I can only assist with questions about Jupiter's banking services. How can I help with your account, payments, or cards? You can also browse through FAQ categories"
            return {
                "response": response_text,
                "source_faqs": [],
                "suggestions": []
            }

        # 2. If on-topic, proceed to generate a full answer
        system_prompt = """You are a helpful customer service assistant for Jupiter, a digital banking app.
Answer the user's question in a friendly, conversational, and accurate manner.
Do not make up facts. If you're unsure, politely say so and suggest contacting support."""
        
        user_prompt = f"User asked: {translated_query}"

        try:
            response = self.llm.invoke([
                ("system", system_prompt),
                ("user", user_prompt)
            ])
            response_text = response.content.strip()
            # Only translate back if the query was on-topic
            translated_back = self._translate_to_original_language(response_text, original_lang)
        except Exception as e:
            logger.error(f"[LLM-Only] LLM error: {e}")
            translated_back = "Sorry, something went wrong while generating the response."

        self.conversation_history.append({
            "query": query,
            "response": translated_back,
            "timestamp": self._get_timestamp()
        })

        return {
            "response": translated_back,
            "source_faqs": [],
            "suggestions": []
        }

    def _get_timestamp(self) -> str:
        return datetime.now().isoformat()

    def get_conversation_history(self):
        return self.conversation_history

    def clear_history(self):
        self.conversation_history = []
        logger.info("[LLM-Only] History cleared.")

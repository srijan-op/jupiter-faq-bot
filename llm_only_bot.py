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

    def get_response(self, query: str) -> dict:
        logger.info(f"[LLM-Only] User Query: {query}")

        original_lang = detect(query)
        translated_query = self._translate_to_english(query)

        if not translated_query.strip():
            return {
                "response": "Please ask a valid question about Jupiter's banking services.",
                "confidence": 0.0,
                "source_faqs": [],
                "suggestions": []
            }

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
            "confidence": 1.0,  # assumed
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

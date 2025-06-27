import pandas as pd
import chromadb
import uuid
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions


class FAQVectorStore:
    def __init__(self, file_path="jupiter_faqs_processed.csv", db_path="vectorstore_faq"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.data.columns = self.data.columns.str.strip().str.lower()  # normalize headers
        self.questions = self.data["question"].tolist()
        self.answers = self.data["answer"].tolist()

        # FIX: use model name string, not the model object
        self.model_name = "all-MiniLM-L6-v2"
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.model_name)

        # Create ChromaDB persistent client and collection
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="faq_collection",
            embedding_function=self.embedding_function
        )


    def load_faqs(self):
        if not self.collection.count():
            for i, question in enumerate(self.questions):
                answer = self.answers[i]
                self.collection.add(
                    documents=[question],
                    metadatas=[{"answer": answer}],
                    ids=[str(uuid.uuid4())]
                )
            print(f"[INFO] Loaded {len(self.questions)} FAQs into ChromaDB.")
        else:
            print("[INFO] FAQ collection already populated.")

    def query_faq(self, user_query, n_results=3):
        results = self.collection.query(
            query_texts=[user_query],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        return results


# ----------------------------------------
# Run the whole process directly
# ----------------------------------------
if __name__ == "__main__":
    faq_db = FAQVectorStore(file_path="jupiter_faqs_processed.csv")
    faq_db.load_faqs()

    # Test query
    query = input("\nAsk a question to test the FAQ search: ")
    result = faq_db.query_faq(query)

    print("\nTop matching FAQs:\n")
    for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
        print(f"Q: {doc}\nA: {meta['answer']}\n---")

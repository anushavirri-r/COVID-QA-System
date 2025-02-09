import nltk
import spacy
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever

# ✅ Load NLP models
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")

# ✅ Load Sentence Transformer Model for Embeddings
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# ✅ Paths
INDEX_PATH = "data/bm25_index.json"
EMBEDDING_PATH = "data/context_embeddings.json"

# ✅ Load Stored BM25 Documents
document_store = InMemoryDocumentStore(use_bm25=True)
retriever = BM25Retriever(document_store=document_store)

with open(INDEX_PATH, "r", encoding="utf-8") as f:
    processed_docs = json.load(f)
document_store.write_documents(processed_docs)

# ✅ Load Context Embeddings
with open(EMBEDDING_PATH, "r", encoding="utf-8") as f:
    context_embeddings = json.load(f)

# ✅ Convert Stored Embeddings to NumPy
context_vectors = {key: np.array(value) for key, value in context_embeddings.items()}

def preprocess_query(query):
    """Preprocess query: lowercase, remove punctuation, tokenize."""
    query = query.lower()
    query = "".join([char for char in query if char.isalnum() or char.isspace()])
    tokens = nltk.word_tokenize(query)
    return " ".join(tokens)

def retrieve_context(query):
    """Retrieve the most relevant context using BM25 & Semantic Similarity."""
    
    # ✅ Step 1: Use BM25 to Get Initial Candidates
    top_docs = retriever.retrieve(query=query, top_k=5)
    top_contents = [doc.content for doc in top_docs]

    # ✅ Step 2: Convert Query to Embedding
    query_embedding = embedding_model.encode(query)

    # ✅ Step 3: Compute Similarity Scores
    similarity_scores = {
        content: np.dot(query_embedding, context_vectors[content]) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(context_vectors[content])
        )
        for content in top_contents
    }

    # ✅ Step 4: Select Most Relevant Context
    best_context = max(similarity_scores, key=similarity_scores.get, default="No relevant context found.")
    
    return best_context

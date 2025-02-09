# import json
# import os
# from haystack.document_stores import InMemoryDocumentStore
# from haystack.nodes import BM25Retriever, PreProcessor
# from haystack.schema import Document  # Import Document for type checking

# # Paths for dataset and saved index
# DATASET_PATH = "data/COVID-QA.json"
# INDEX_PATH = "data/bm25_index.json"  # Save processed documents

# def load_data(filepath):
#     """Load dataset from JSON format."""
#     with open(filepath) as f:
#         data = json.load(f)

#     documents = []
#     for article in data["data"]:
#         for paragraph in article["paragraphs"]:
#             context_text = paragraph.get("context", "No context available")  
#             title = context_text.split("\n")[0] if context_text else "Unknown"
#             documents.append({"content": context_text, "meta": {"source": title}})  
#     return documents

# # ✅ Initialize InMemoryDocumentStore with BM25 enabled
# document_store = InMemoryDocumentStore(use_bm25=True)

# # ✅ Check if processed documents already exist
# if os.path.exists(INDEX_PATH):
#     print("✅ Loading preprocessed documents...")
#     with open(INDEX_PATH) as f:
#         processed_docs = json.load(f)
# else:
#     print("⚡ Processing documents...")
#     documents = load_data(DATASET_PATH)
#     preprocessor = PreProcessor(split_by="word", split_length=200, split_overlap=20)
#     processed_docs = preprocessor.process(documents)

#     # ✅ Convert Document objects to dictionaries before saving
#     processed_docs_json = [
#         {"content": doc.content, "meta": doc.meta} if isinstance(doc, Document) else doc
#         for doc in processed_docs
#     ]

#     with open(INDEX_PATH, "w", encoding="utf-8") as f:
#         json.dump(processed_docs_json, f, ensure_ascii=False, indent=2)

# # ✅ Write to document store
# document_store.write_documents(processed_docs)

# print("✅ Context preparation completed. BM25 index is ready!")

import json
import os
from sentence_transformers import SentenceTransformer
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever

# ✅ Load Sentence Transformer Model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# ✅ Paths
DATASET_PATH = "data/COVID-QA.json"
INDEX_PATH = "data/bm25_index.json"
EMBEDDING_PATH = "data/context_embeddings.json"

# ✅ Load dataset
def load_data(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    documents = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            documents.append({"content": paragraph["context"]})
    return documents

# ✅ Initialize Document Store
document_store = InMemoryDocumentStore(use_bm25=True)

# ✅ Load & Index Data
documents = load_data(DATASET_PATH)
document_store.write_documents(documents)

# ✅ Save BM25 Index
with open(INDEX_PATH, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

# ✅ Compute & Store Embeddings for Semantic Retrieval
context_embeddings = {}
for doc in documents:
    context_embeddings[doc["content"]] = embedding_model.encode(doc["content"]).tolist()

# ✅ Save Embeddings
with open(EMBEDDING_PATH, "w", encoding="utf-8") as f:
    json.dump(context_embeddings, f, ensure_ascii=False, indent=2)

print("✅ Context preparation completed. Embeddings stored successfully!")

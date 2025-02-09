# import json
# import os
# import nltk
# import spacy
# from googletrans import Translator  # ✅ Using Google Translate API
# from langdetect import detect
# from transformers import pipeline
# from query_processing import preprocess_query, retrieve_context  # Import from query_processing

# # Load NLP models
# nltk.download("averaged_perceptron_tagger")
# nltk.download("punkt")
# nlp = spacy.load("en_core_web_sm")

# # ✅ Load Pretrained Transformer for Question Answering
# qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# # ✅ Initialize Google Translator
# translator = Translator()

# def detect_language(text):
#     """Detect the language of the query."""
#     return detect(text)

# def translate_text(text, target_lang="en"):
#     """Translate text to the target language (default: English)."""
#     source_lang = detect_language(text)
#     if source_lang == target_lang:
#         return text, source_lang  # No translation needed
#     translated_text = translator.translate(text, src=source_lang, dest=target_lang).text
#     return translated_text, source_lang  # Return translated text and original language

# def extract_answer(query):
#     """Retrieve relevant context and extract an answer using a QA model."""
#     # ✅ Detect language and translate query to English
#     translated_query, original_lang = translate_text(query)
#     # ✅ Process the query
#     processed_query = preprocess_query(translated_query)

#     # ✅ Retrieve relevant context from stored BM25 index
#     relevant_context = retrieve_context(processed_query)

#     # ✅ Print retrieved context to verify correctness
#     print("\n--- Retrieved Context ---")
#     print(relevant_context)
#     print("-------------------------")

#     # ✅ Ensure context is valid before extracting the answer
#     if relevant_context == "No relevant context found.":
#         return "⚠️ No relevant context available to extract an answer."

#     # ✅ Extract the answer using the QA model
#     qa_result = qa_pipeline(question=translated_query, context=relevant_context)
#     extracted_answer = qa_result["answer"]

#     # ✅ Translate answer back to original language if needed
#     if original_lang != "en":
#         extracted_answer = translator.translate(extracted_answer, src="en", dest=original_lang).text

#     return extracted_answer

# # Example Usage
# if __name__ == "__main__":
#     user_query = input("Enter your question: ")
#     answer = extract_answer(user_query)
    
#     # Print the extracted answer
#     print(f"\nExtracted Answer: {answer}")
import json
import os
import nltk
import spacy
import numpy as np
from googletrans import Translator
from langdetect import detect
from transformers import pipeline
from query_processing import preprocess_query, retrieve_context  # Import functions

# ✅ Load NLP models
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")

# ✅ Load Pretrained Transformer for Question Answering
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# ✅ Initialize Google Translator
translator = Translator()

def detect_language(text):
    """Detect the language of the query."""
    return detect(text)

def translate_text(text, target_lang="en"):
    """Translate text to the target language (default: English)."""
    source_lang = detect_language(text)
    if source_lang == target_lang:
        return text, source_lang  # No translation needed
    translated_text = translator.translate(text, src=source_lang, dest=target_lang).text
    return translated_text, source_lang  # Return translated text and original language

def extract_answer(query):
    """Retrieve relevant context and extract an answer using a QA model."""
    # ✅ Detect language and translate query to English
    translated_query, original_lang = translate_text(query)

    # ✅ Process the query
    processed_query = preprocess_query(translated_query)

    # ✅ Retrieve relevant context using BM25 & Semantic Similarity
    relevant_context = retrieve_context(processed_query)

    # ✅ Print retrieved context for debugging
    print("\n--- Retrieved Context ---")
    print(relevant_context)
    print("-------------------------")

    # ✅ Ensure valid context before proceeding
    if relevant_context == "No relevant context found.":
        return "⚠️ No relevant context available to extract an answer."

    # ✅ Extract answer using the QA model
    qa_result = qa_pipeline(question=translated_query, context=relevant_context)
    extracted_answer = qa_result["answer"]
    print(extracted_answer)
    # ✅ Translate answer back to original language if needed
    if original_lang != "en":
        extracted_answer = translator.translate(extracted_answer, src="en", dest=original_lang).text

    return extracted_answer

# ✅ Example Usage
if __name__ == "__main__":
    user_query = input("Enter your question: ")
    answer = extract_answer(user_query)
        
    # ✅ Print the extracted answer
    print(f"\nExtracted Answer: {answer}")

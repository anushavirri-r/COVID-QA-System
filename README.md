# 🚀 COVID-QA-System  
A **Multilingual Question Answering (QA) System** for COVID-19 using **BM25, Dense Embeddings, and Advanced NLP Techniques**. Supports **English, Telugu, Hindi, Spanish, and more!**  

![GitHub Repo Size](https://img.shields.io/github/repo-size/anushavirri-r/COVID-QA-System?style=flat-square)  
![GitHub Stars](https://img.shields.io/github/stars/anushavirri-r/COVID-QA-System?style=flat-square)  
![GitHub Forks](https://img.shields.io/github/forks/anushavirri-r/COVID-QA-System?style=flat-square)  

---

## 📌 Features  
✅ **Multilingual Support** – Detects query language & translates automatically  
✅ **Advanced NLP Techniques** – POS Tagging, Named Entity Recognition (NER), Dependency Parsing  
✅ **Hybrid Retrieval** – Combines **BM25** (Lexical) & **Dense Embeddings** (Semantic)  
✅ **Pretrained QA Model** – Uses `roberta-base-squad2` for precise answers  
✅ **Optimized Storage** – Saves embeddings for faster retrieval  
✅ **Google Translate API** – Translates both **queries** & **answers**  

---

## 📂 Project Structure  
COVID-QA-System/ │── data/ # Stores BM25 Index & Context Embeddings │ ├── COVID-QA.json # COVID QA dataset │ ├── bm25_index.json # Preprocessed BM25 index │ ├── context_embeddings.json # Saved embeddings for semantic retrieval │ │── context_preparation.py # Loads dataset, processes context, stores embeddings │── query_processing.py # Preprocesses queries & retrieves context using NLP │── answer_extraction.py # Extracts answer from retrieved context │── requirements.txt # List of dependencies │── README.md # Project documentation

---

## 📦 Installation  

### 🔹 Step 1: Clone the Repository  
```bash
git clone https://github.com/anushavirri-r/COVID-QA-System.git
cd COVID-QA-System

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

## 📦 Installation  

### 🔹 Step 1: Clone the Repository  
```bash
git clone https://github.com/anushavirri-r/COVID-QA-System.git
cd COVID-QA-System
```
### 🔹 Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🛠️ Technologies Used  
- **NLP & Transformers:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), [spaCy](https://spacy.io/), [nltk](https://www.nltk.org/)  
- **Retrieval Models:** [BM25](https://en.wikipedia.org/wiki/Okapi_BM25), [Sentence Transformers (MiniLM)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
- **Machine Learning Frameworks:** [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers/index)  
- **Translation:** [Google Translate API](https://cloud.google.com/translate)  
- **Data Storage:** JSON

---

## 💡 Future Improvements  
- 🔹 Support more datasets beyond COVID-QA  
- 🔹 Improve answer ranking using Re-rankers  
- 🔹 Deploy as a REST API using FastAPI  

# 🤖 Arokyamary's AI Assistant

A RAG-based AI chatbot that answers business questions from CSV data using LangChain, ChromaDB, and Groq LLaMA 3.

## 🔗 Live Demo
https://p2-rag-chatbot-a8pgr8g4w3ezpj6tkmfhcn.streamlit.app

## 🛠️ Tech Stack
- **LangChain** — RAG pipeline
- **ChromaDB** — Vector store
- **HuggingFace** — Embedding model (all-MiniLM-L6-v2)
- **Groq LLaMA 3** — Free LLM API
- **Streamlit** — Chat UI and deployment

## 📊 Features
- Answers business questions from sales and product CSV data
- Handles general questions like a normal chatbot
- Sample questions in sidebar for quick testing
- Deployed live on Streamlit Cloud

## Project Structure

```
P2_RAG_Chatbot/
├── app.py                    # Streamlit chat UI
├── rag_engine.py             # RAG pipeline
├── build_vectorstore.py      # Build ChromaDB vectors
├── data/
│   ├── sales_report.csv
│   └── product_data.csv
├── requirements.txt
└── .gitignore
```

## 🚀 Run Locally
```bash
git clone https://github.com/Arokyamary/P2-RAG-Chatbot.git
cd P2-RAG-Chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 👩‍💼 Built by Arokyamary

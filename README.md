# 📚 PDF Genie 🤖✨  
Turn any PDF into an interactive chatbot powered by Retrieval-Augmented Generation (RAG).

---

## 🚀 Features
- Upload any PDF and chat with it instantly.  
- Smart text chunking (sentence/paragraph based).  
- Vector search powered by **ChromaDB**.  
- Embeddings via **Gemini / HuggingFace SentenceTransformers**.  
- Streamlit UI with adjustable retrieval settings (Top-K, min relevance).  
- Citations included (with toggle to show/hide).  
- Lightweight & customizable.

---

## 🗂 Project Structure
```
PDF-RAG-Chatbot/
│── app/
│   ├── chunkers/        # Text chunking logic
│   ├── embed/           # Embedding models
│   ├── llm/             # LLM integration
│   ├── vector/          # Vector DB (Chroma)
│   ├── ingestion.py     # PDF ingestion & indexing
│   ├── query.py         # Query handling (retrieval + generation)
│   └── ...
│
│── data/
│   ├── index/           # Persistent Chroma index (ignored in git)
│   └── raw_pdfs/        # Uploaded PDFs (ignored in git)
│
│── ui/
│   └── streamlit_app.py # Streamlit interface
│
│── requirements.txt
│── .gitignore
│── README.md
```

---

## ⚡️ Installation

### 1. Clone repo
```bash
git clone https://github.com/SaadSaddique/pdf-genie.git
cd pdf-genie
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup environment variables
Create a `.env` file in root:
```
OPENAI_API_KEY=your_api_key_here
# or Gemini key if you’re using Gemini embeddings
```

---

## ▶️ Usage

Run the Streamlit app:
```bash
streamlit run ui/streamlit_app.py
```

Then open the local URL shown (default: `http://localhost:8501`).

---

## 🔧 Retrieval Settings
- **Top-K**: Number of chunks to retrieve (higher = broader, slower).  
- **Min Relevance**: Filter irrelevant chunks (lower value = stricter).  
- **Chunker**: Sentence / token based splitting.  
- **Chunk Size & Overlap**: Controls context length and redundancy.  

> 💡 Suggestions are provided in the Streamlit app based on PDF size.

---

## 📌 Example
1. Upload your PDF.  
2. Click **Index Now**.  
3. Ask natural language questions like:  
   - *"Summarize this PDF"*  
   - *"What are the key findings?"*  
   - *"Explain section 3 in simple terms"*  

---

## 🤝 Contributing
PRs and issues are welcome! Feel free to fork and improve.

---

## 📜 License
[MIT](LICENSE)

---

## 👨‍💻 Author
Built with ❤️ by [Your Name](https://github.com/SaadSaddique)  

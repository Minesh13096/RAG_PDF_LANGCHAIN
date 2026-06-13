# 📄 PDF RAG Chatbot

A simple **end-to-end RAG (Retrieval-Augmented Generation)** app. Upload a PDF,
ask questions, and get answers grounded in the document.

**Stack:** LangChain · FAISS · HuggingFace (free tier) · Streamlit

---

## What is RAG? (the 7 steps used in this app)

| Step | What happens | Tool |
|------|--------------|------|
| 1. Load | Read the PDF into text | `PyPDFLoader` |
| 2. Chunk | Split text into small overlapping pieces | `RecursiveCharacterTextSplitter` |
| 3. Embed | Turn each chunk into a vector (numbers) | `sentence-transformers/all-MiniLM-L6-v2` |
| 4. Store | Save vectors in a searchable database | `FAISS` |
| 5. Retrieve | Find chunks most similar to the question | FAISS retriever |
| 6. Augment | Put those chunks into the prompt as context | prompt building |
| 7. Generate | LLM answers using only that context | `Qwen/Qwen2.5-7B-Instruct` |

Why RAG? The LLM only sees the relevant parts of *your* PDF, so answers are
grounded in the document instead of made up.

---

## 1. Run it locally

```bash
# clone your repo
git clone https://github.com/<your-username>/rag-pdf-chatbot.git
cd rag-pdf-chatbot

# create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

Get a free HuggingFace token at <https://huggingface.co/settings/tokens>
(type "Read" is enough). Then create your local secrets file:

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# open .streamlit/secrets.toml and paste your real hf_... token
```

Run the app:

```bash
streamlit run app.py
```

It opens at <http://localhost:8501>. Upload a PDF and start asking questions.

---

## 2. Put it on GitHub

```bash
git init
git add .
git commit -m "Initial commit: RAG PDF chatbot"
git branch -M main
git remote add origin https://github.com/<your-username>/rag-pdf-chatbot.git
git push -u origin main
```

> ✅ `.streamlit/secrets.toml` is in `.gitignore`, so your token is **not** pushed.

---

## 3. Deploy on Streamlit Cloud (free)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** → pick your `rag-pdf-chatbot` repo, branch `main`, file `app.py`.
3. Click **Advanced settings → Secrets** and paste:
   ```toml
   HUGGINGFACEHUB_API_TOKEN = "hf_your_token_here"
   ```
4. Click **Deploy**. First build takes a few minutes (it downloads the
   embedding model). After that you get a public URL to share.

---

## Notes & troubleshooting

- **First question is slow:** the embedding model downloads on first run, then it's cached.
- **Free tier limits:** HuggingFace gives a monthly credit allowance for the LLM. Fine for demos.
- **Model not available error:** swap the `repo_id` in `app.py` to
  `meta-llama/Llama-3.2-3B-Instruct` or `mistralai/Mistral-7B-Instruct-v0.3`.
- **Memory limit on Streamlit Cloud:** if the app runs out of memory, switch the
  embeddings to the API version (`HuggingFaceEndpointEmbeddings`) so `torch`
  isn't loaded — ask if you want that variant.

---

## Project structure

```
rag-pdf-chatbot/
├── app.py                          # the whole RAG app
├── requirements.txt                # dependencies
├── README.md                       # this file
├── .gitignore
└── .streamlit/
    └── secrets.toml.example        # token template (copy to secrets.toml)
```

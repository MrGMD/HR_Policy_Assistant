# 📘 HR Policy Assistant — RAG with Streamlit

An AI-powered **HR Policy Assistant** that allows users to upload an HR policy PDF and ask questions about it.

The application uses **Retrieval-Augmented Generation (RAG)** so that answers are generated from the uploaded policy rather than relying only on the language model's general knowledge.

## Features

- Upload an HR policy PDF
- Extract PDF text using **PyMuPDF**
- Split the policy into overlapping chunks
- Create embeddings using **Sentence Transformers**
- Store embeddings in **FAISS**
- Retrieve the most relevant policy sections for each question
- Generate answers using **Groq**
- Show source page numbers and retrieved text
- Chat-style Streamlit interface
- Works locally and can be deployed through **GitHub + Streamlit Community Cloud**
- No HR policy documents are stored permanently by the application

## RAG Architecture

```text
                ┌──────────────────┐
                │   HR Policy PDF  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    PyMuPDF       │
                │   Text Extract   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Text Chunking    │
                │ + Overlap        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Sentence         │
                │ Transformers     │
                │ Embeddings       │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │      FAISS       │
                │ Vector Index     │
                └────────┬─────────┘
                         │
User Question ──────────►│
                         ▼
                ┌──────────────────┐
                │ Relevant Chunks  │
                │     Top-K        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │      Groq        │
                │ LLM Generation   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ HR Policy Answer │
                │ + Source Pages  │
                └──────────────────┘
```

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Web interface |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| Groq | LLM response generation |
| NumPy | Embedding/vector processing |

## Project Structure

```text
hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/hr-policy-assistant.git
cd hr-policy-assistant
```

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

If PowerShell blocks activation, you can use:

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

and run Streamlit with:

```powershell
venv\Scripts\python.exe -m streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Groq API Key

Get a Groq API key from your Groq account.

### Local development

You can enter the API key directly into the sidebar when the application starts.

Or set an environment variable.

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

### macOS/Linux

```bash
export GROQ_API_KEY="your_groq_api_key"
```

Do **not** put your API key directly inside `app.py`.

## 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 6. Use the Application

1. Open the application.
2. Enter your Groq API key if it is not already configured.
3. Upload an HR policy PDF.
4. Wait for the document to be processed.
5. Ask a question such as:
   - "How many annual leave days are employees entitled to?"
   - "What is the maternity leave policy?"
   - "Who approves overtime?"
   - "What are the working hours?"
   - "What is the probation period?"
6. The application retrieves relevant sections and sends only those sections to the Groq model.
7. The response includes the relevant source pages.

## Deploy on GitHub + Streamlit Community Cloud

### Step 1 — Create a GitHub repository

Create a new repository, for example:

```text
hr-policy-assistant
```

Do not upload your API key.

### Step 2 — Add the project files

Upload:

```text
app.py
requirements.txt
README.md
.gitignore
```

### Step 3 — Push using Git

```bash
git init
git add .
git commit -m "Initial HR Policy Assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hr-policy-assistant.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 4 — Deploy with Streamlit Community Cloud

1. Open Streamlit Community Cloud.
2. Sign in with GitHub.
3. Create a new app.
4. Select your GitHub repository.
5. Select the `main` branch.
6. Set the main file to:

```text
app.py
```

7. Deploy the application.

### Step 5 — Add the Groq Secret

In your Streamlit app settings, open **Secrets** and add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Save the secret and restart/redeploy the application.

The application reads this value automatically.

## Important Security Rules

Never commit any of these to GitHub:

```text
.env
.env.local
.streamlit/secrets.toml
```

Never put this in `app.py`:

```python
GROQ_API_KEY = "your-real-api-key"
```

Use Streamlit Secrets or an environment variable instead.

## How the RAG System Works

### 1. PDF ingestion

PyMuPDF extracts readable text from every page.

Each extracted page keeps its page number so that the application can show source references.

### 2. Chunking

Long policy text is divided into smaller overlapping chunks.

The current configuration is:

```text
Chunk size: 800 words
Overlap:    120 words
```

Overlap helps preserve context when important information crosses chunk boundaries.

### 3. Embeddings

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Each chunk is converted into a numerical vector.

### 4. FAISS

FAISS stores the vectors and performs similarity search.

For every question, the application retrieves the top 5 most relevant chunks.

### 5. Groq

The retrieved policy chunks are passed to the Groq LLM.

The prompt instructs the model to:

- use only the supplied policy context
- avoid inventing information
- mention page numbers when useful
- identify when information is not present
- preserve conditions and exceptions

## Limitations

### Scanned PDFs

This version works best with text-based PDFs.

If a PDF contains scanned images instead of selectable text, PyMuPDF may not extract meaningful text. OCR would need to be added for those documents.

### Session-based indexing

The FAISS index is created in the current Streamlit session.

If the application restarts, the uploaded document needs to be uploaded and indexed again.

This is intentional for a simple GitHub/Streamlit deployment.

### Policy accuracy

The assistant should be treated as a policy search and question-answering tool, not as a replacement for HR or legal professionals.

For sensitive employment decisions or legal interpretation, verify the answer against the official policy and consult the appropriate HR/legal professional.

## Customization

You can customize the following values in `app.py`:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5
```

You can also change the Groq model in:

```python
model="llama-3.3-70b-versatile"
```

## Future Improvements

Possible production improvements include:

- OCR for scanned HR policy PDFs
- support for multiple PDFs
- persistent FAISS indexes
- document management
- user authentication
- conversation history
- policy versioning
- department-specific policies
- access control
- answer confidence thresholds
- hybrid keyword + vector search
- reranking
- document metadata filtering
- policy comparison between versions
- admin dashboard
- audit logs

## License

This project can be used as a starting point for educational, portfolio, or internal HR-policy applications. Add an appropriate license before distributing it publicly.

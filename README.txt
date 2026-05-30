# Employee RAG System

An AI-powered Streamlit application that allows users to upload PDF documents, extract text, build a FAISS vector store, and ask questions based on the uploaded content. The app also generates structured learning courses and quiz questions from the documents using Groq LLM models.

## Features

- Upload multiple PDF files.
- Extract text from PDFs with `pdfplumber`.
- Split document text into chunks for retrieval.
- Create embeddings using Sentence Transformers.
- Store embeddings in FAISS.
- Ask questions based only on uploaded documents.
- Generate a structured course from the uploaded PDFs.
- Show quizzes and track progress.
- Create a training progress PDF report.

## Project Structure

```text
employee-rag-system/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml   # local only, do not upload
```

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq API
- FAISS
- Sentence Transformers
- pdfplumber
- reportlab

## Requirements

- Python 3.10 or later
- A Groq API key
- Internet access for installing packages and downloading embedding models

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add your Groq API key

Create this file locally:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Do not commit this file to GitHub.

### 6. Run the app

```bash
streamlit run app.py
```

## How to Use

1. Open the app in your browser.
2. Upload one or more PDF documents.
3. Select your role and learning focus.
4. Ask a question about the uploaded files.
5. Generate a course from the uploaded content.
6. Review the quiz and progress report features.

## Deployment on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from your GitHub repository.
4. Set the main file path to `app.py`.
5. Open **Advanced settings**.
6. Paste your `GROQ_API_KEY` into the Secrets box.
7. Deploy the app.

## Important Notes

- The app only answers questions using the uploaded documents.
- The vector store is created after PDFs are uploaded and processed.
- All secrets should stay outside GitHub.
- Make sure `requirements.txt` includes every package used by the app.

## Author

Arjuna Krishan
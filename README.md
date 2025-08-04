# Multimodal Rag System

The Multimodal RAG System enables users to get precise answers to their questions by retrieving information from specific documents. Unlike traditional text-only retrieval systems, it can seamlessly process text, images, and tables, delivering rich, context-aware responses.

## How to run the system

You must download these applications first

1) Download Poppler binaries from:

```bash
https://github.com/oschwartz10612/poppler-windows/releases
```

2) Install Tesseract:

```bash
https://github.com/UB-Mannheim/tesseract/wiki
```

5) Use venv:

```bash
python -m venv .venv
.venv\Scripts\activate
```
4) Install requirements:

```bash
pip install -r requirements.txt
```

5) Create a ```.env``` file in the root folder:
```bash
GOOGLE_API_KEY = <GOOGLE_API_KEY>
```

6) Run the app:

```bash
 streamlit run main.py
```

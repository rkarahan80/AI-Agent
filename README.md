# Simple RAG Chatbot

This project implements a command-line Retrieval Augmented Generation (RAG) chatbot using Python, Langchain, and OpenAI.

## Features

- Answers questions based on the content of `training_data.txt`.
- Uses OpenAI for language understanding and generation.
- Uses OpenAI embeddings and a local vector store (ChromaDB or FAISS) for efficient retrieval.

## Setup and Usage

### 1. Prerequisites
- Python 3.7+
- Access to an OpenAI API key.

### 2. Environment Variables
You **must** set your OpenAI API key as an environment variable.

**Linux/macOS:**
```bash
export OPENAI_API_KEY='your_actual_openai_api_key_here'
```

**Windows (PowerShell):**
```powershell
$Env:OPENAI_API_KEY='your_actual_openai_api_key_here'
```
Replace `your_actual_openai_api_key_here` with your real key. You might want to add this line to your shell's profile file (e.g., `.bashrc`, `.zshrc`, or PowerShell profile) for persistence.

### 3. Training Data
- Locate the `training_data.txt` file in the root directory of this project.
- **Replace the placeholder content in `training_data.txt` with your own text data.** This data will be used by the chatbot to answer questions. The quality and relevance of this data are crucial for the chatbot's performance.
- Ensure the file is saved with UTF-8 encoding if you are adding non-ASCII characters.

### 4. Running the Chatbot
Once the `OPENAI_API_KEY` is set and `training_data.txt` is populated, you can run the chatbot:

```bash
python rag_chatbot.py
```

The script will:
1. Load and process your data from `training_data.txt`.
2. Initialize the RAG pipeline (this might take a moment, especially the first time when embeddings are generated).
3. Prompt you to enter your questions.

Type your question and press Enter. Type `exit` or `quit` to close the chatbot.

### 5. Dependencies
The script uses the following major Python libraries:
- `langchain`
- `openai`
- `langchain_community` (for TextLoader, Chroma, FAISS)
- `tiktoken` (tokenizer for OpenAI models)
- `faiss-cpu` (optional, if Chroma fails or for FAISS vector store)
- `chromadb` (optional, for Chroma vector store)

These dependencies are listed in `requirements.txt`. Due to potential environment limitations during automated setup, a virtual environment might not have been created automatically. If you encounter import errors, you may need to install these packages manually in your Python environment:
```bash
pip install langchain openai langchain_community tiktoken faiss-cpu chromadb
```
(Note: `faiss-cpu` and `chromadb` are the primary vector store options the script tries. You might only need one depending on what works in your environment.)

---
# AI-Agent
# Simple RAG Chatbot

This project implements a command-line Retrieval Augmented Generation (RAG) chatbot using Python, Langchain, and OpenAI.

## Features

- Answers questions based on the content of `training_data.txt`.
- Uses OpenAI for language understanding and generation (by default).
- Uses OpenAI embeddings (by default) and a local vector store (ChromaDB or FAISS) for efficient retrieval.
- Pluggable AI model architecture (currently supporting OpenAI, with placeholders for Google Jules and MCP).

## Setup and Usage

### 1. Prerequisites
- Python 3.7+
- Access to an OpenAI API key.

### 2. Environment Variables
You need to configure environment variables to specify your chosen AI model provider and the necessary API keys.

-   **`OPENAI_API_KEY`**: Required if using the OpenAI model (default). Replace `your_actual_openai_api_key_here` with your real key.
-   **`AI_MODEL_PROVIDER`**: Specifies which AI model to use.
    -   Supported values: `OPENAI` (default), `JULES`, `MCP`.
    -   Example: `export AI_MODEL_PROVIDER='OPENAI'`
-   **`JULES_API_KEY`**: (Future Use) Will be required if `AI_MODEL_PROVIDER` is set to `JULES` once the model is implemented. For now, a dummy value is used if not set.
-   **`MCP_API_KEY`**: (Future Use) Will be required if `AI_MODEL_PROVIDER` is set to `MCP` once the model is implemented. For now, a dummy value is used if not set.


**Linux/macOS:**
```bash
export OPENAI_API_KEY='your_actual_openai_api_key_here'
export AI_MODEL_PROVIDER='OPENAI' # Or 'JULES', 'MCP'
# export JULES_API_KEY='your_jules_key_here' # When Jules is implemented
# export MCP_API_KEY='your_mcp_key_here'     # When MCP is implemented
```

**Windows (PowerShell):**
```powershell
$Env:OPENAI_API_KEY='your_actual_openai_api_key_here'
$Env:AI_MODEL_PROVIDER='OPENAI' # Or 'JULES', 'MCP'
# $Env:JULES_API_KEY='your_jules_key_here' # When Jules is implemented
# $Env:MCP_API_KEY='your_mcp_key_here'     # When MCP is implemented
```
You might want to add these lines to your shell's profile file (e.g., `.bashrc`, `.zshrc`, or PowerShell profile) for persistence.

## AI Model Configuration

This application supports a pluggable AI model architecture, allowing you to choose from different AI providers. The selection is controlled by the `AI_MODEL_PROVIDER` environment variable.

-   **`AI_MODEL_PROVIDER`**:
    -   `OPENAI`: Uses OpenAI's models for embeddings and language generation. Requires `OPENAI_API_KEY`. This is the default if the variable is not set.
    -   `JULES`: Placeholder for a hypothetical "Jules" AI model. Currently, this is a non-functional placeholder. If selected, the application will indicate this and will not be able to process queries.
    -   `MCP`: Placeholder for a hypothetical "MCP" AI model. Similar to `JULES`, this is a non-functional placeholder.

The core abstraction is defined in `ai_models.py` with the `AIModel` base class. Developers can extend this by adding new classes that implement the required methods (`get_embeddings`, `generate_response`) for other AI services.

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
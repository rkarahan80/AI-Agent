# RAG Chatbot with Agent Development Kit (ADK)

This project implements a command-line and web-ui enabled Retrieval Augmented Generation (RAG) chatbot using Python, Langchain, OpenAI, and Google's Agent Development Kit (ADK).

## Features

- Answers questions based on the content of `training_data.txt`.
- Uses OpenAI for language understanding and generation, integrated via Langchain.
- Leverages Google's Agent Development Kit for agent structure and execution.
- Provides interaction via command-line (`adk run`) or a web UI (`adk web`).
- Uses OpenAI embeddings and a local vector store (ChromaDB or FAISS) for efficient retrieval.

## Setup and Usage

### 1. Prerequisites
- Python 3.7+
- Access to an OpenAI API key.
- Git (for cloning the repository).

### 2. Clone the Repository
If you haven't already, clone this repository to your local machine.

### 3. Install Dependencies
Navigate to the project's root directory in your terminal and install the required Python packages:
```bash
pip install -r requirements.txt
```
This will install all necessary libraries, including `langchain`, `openai`, `google-adk`, `litellm`, and others.

### 4. Set Up OpenAI API Key
The agent requires an OpenAI API key to function. You need to place this key in a dedicated `.env` file:
   - Navigate to the `rag_adk_agent` directory within the project.
   - Create a file named `.env` (if it doesn't already exist).
   - Add the following line to the `.env` file, replacing `YOUR_ACTUAL_OPENAI_API_KEY_HERE` with your real key:
     ```
     OPENAI_API_KEY="YOUR_ACTUAL_OPENAI_API_KEY_HERE"
     ```
   - **Important**: Ensure this file is saved and the key is correct. Do not commit your API key to version control if you're managing this project with Git. The `.gitignore` file should ideally list `rag_adk_agent/.env`.

### 5. Prepare Training Data
- Locate the `training_data.txt` file in the root directory of this project.
- **Replace the placeholder content in `training_data.txt` with your own text data.** This data will be used by the chatbot to answer questions. The quality and relevance of this data are crucial for the chatbot's performance.
- Ensure the file is saved with UTF-8 encoding if you are adding non-ASCII characters.

### 6. Running the Chatbot with ADK

Once the dependencies are installed, the API key is set in `rag_adk_agent/.env`, and `training_data.txt` is populated, you can run the chatbot using ADK commands from the **root directory** of the project:

**Option 1: Command-Line Interface**
```bash
adk run rag_adk_agent
```
This command will:
1. Load the ADK agent defined in the `rag_adk_agent` directory.
2. Initialize the RAG pipeline (this might take a moment, especially the first time when embeddings for your data are generated).
3. Provide a command-line prompt for you to enter questions.
Type your question and press Enter. Type `exit` or `quit` to close the chatbot.

**Option 2: Web User Interface**
```bash
adk web
```
This command will:
1. Start a local web server (usually at `http://localhost:8000` or `http://127.0.0.1:8000`).
2. Open the ADK development UI in your web browser.
3. In the UI, select "rag_chatbot_agent" from the agent dropdown list.
4. You can then interact with the chatbot through the web interface, which may also offer options to inspect agent events and traces.
Follow the on-screen instructions to interact. To stop the server, press `Ctrl+C` in the terminal where you ran `adk web`.

## Running the Streamlit GUI Chatbot

This project also includes a standalone graphical user interface (GUI) for the RAG chatbot, built with Streamlit. This provides an alternative way to interact with the chatbot, now with support for multiple Large Language Models (LLMs).

**Prerequisites:**

1.  **Install Dependencies:**
    Ensure all dependencies, including Streamlit and new LLM provider packages (like `langchain-google-genai`, `langchain-anthropic`, `langchain-deepseek`), are installed by running the following command from the project root:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set Up API Keys:**
    The Streamlit application requires API keys for the LLMs you intend to use. These must be set as environment variables. You can set them in your terminal session (e.g., `export OPENAI_API_KEY='your_key_here'`) or by creating a `.env` file in the project root, which the script attempts to load for local development.
    Required environment variables for the supported models are:
    -   `OPENAI_API_KEY`: For OpenAI models.
    -   `GOOGLE_API_KEY`: For Google Gemini models.
    -   `ANTHROPIC_API_KEY`: For Anthropic Claude models.
    -   `DEEPSEEK_API_KEY`: For Deepseek models.

    **Important for Embeddings:** By default, this application uses OpenAI's text embeddings for document processing. Therefore, `OPENAI_API_KEY` must be set and valid, even if you select a different provider (like Gemini, Claude, or Deepseek) for generating chat responses.

**Running the GUI:**

1.  Navigate to the project's root directory in your terminal.
2.  Run the Streamlit application using:
    ```bash
    streamlit run gui_chatbot_streamlit.py
    ```
3.  This will typically open the chatbot interface in your default web browser. If it doesn't open automatically, the terminal output will provide a local URL (usually `http://localhost:8501`) that you can navigate to.

**Selecting the LLM Provider:**
Once the application is running, you can choose your desired LLM provider (OpenAI, Gemini, Claude, or Deepseek) from the dropdown menu in the sidebar under "⚙️ Configuration". The RAG pipeline will re-initialize with the selected model. Ensure the corresponding API key for your chosen model is correctly set in your environment.

---
# AI-Agent

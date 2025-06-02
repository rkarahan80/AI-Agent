# AI Agent with Document Processing Capabilities

This project implements an AI agent capable of ingesting documents from various sources, processing them using Google Cloud AI services (Document AI, Vertex AI), and archiving the results. The agent is built using Python and LangChain, allowing for natural language interaction to perform its tasks.

## Key Features

*   **Multi-Source Document Ingestion:**
    *   Reads local text files (`.txt`).
    *   Extracts text from local PDF files (`.pdf`) using `pdfplumber`.
    *   Fetches and parses textual content from web pages.
*   **Google Cloud AI Integration:**
    *   Utilizes **Document AI** for OCR and potentially structured data extraction from documents.
    *   Leverages **Vertex AI** for generating text embeddings.
*   **Archiving and Storage:**
    *   Stores original documents, extracted text, and structured metadata in **Google Cloud Storage**.
*   **LangChain-Powered Agent:**
    *   Uses LangChain to create an agent that can understand natural language queries.
    *   Dynamically selects appropriate tools (ingestion, processing, AI services) based on user requests.
    *   Provides an interactive command-line interface for user interaction.
*   **Comprehensive Documentation:**
    *   Detailed guides for setup and usage.

## Getting Started

1.  **Setup and Configuration:**
    *   To set up your Google Cloud environment, configure service accounts, enable APIs, and understand the necessary credentials, please refer to the [**SETUP_GUIDE.md**](SETUP_GUIDE.md).
2.  **User Guide:**
    *   For detailed instructions on installing dependencies, setting environment variables, running the agent, and example interactions, please see the [**USER_GUIDE.md**](USER_GUIDE.md).

## Core Technologies

*   Python
*   LangChain
*   Google Cloud Platform:
    *   Vertex AI
    *   Document AI
    *   Cloud Storage
*   `pdfplumber` for PDF text extraction.
*   `requests` and `BeautifulSoup4` for web content fetching.

## Project Structure (Key Files)

*   `langchain_agent.py`: Main script to run the LangChain-based AI agent.
*   `agent_core.py`: Provides a more direct command-line interface to core functionalities (less natural language dependent).
*   `gcp_ai_services.py`: Contains functions for interacting with Google Cloud AI (Vertex AI, Document AI) and Google Cloud Storage.
*   `document_ingestion.py`: Includes functions for reading/fetching documents from different sources.
*   `SETUP_GUIDE.md`: Guide for setting up the Google Cloud environment.
*   `USER_GUIDE.md`: Comprehensive guide for users on how to run and interact with the agent.
*   `requirements.txt`: Lists all Python dependencies.

# AI Agent User Guide

This guide explains how to set up and use the AI Agent, which is capable of document ingestion, processing via Google Cloud AI services, and archiving.

## 1. Prerequisites

*   Python 3.8 or higher installed.
*   Google Cloud SDK (`gcloud` command-line tool) installed and initialized. (Link to Google Cloud SDK installation guide: https://cloud.google.com/sdk/docs/install)

## 2. Setup and Configuration

For detailed environment setup, Google Cloud Project configuration, API enablement, and service account creation, please follow the instructions in [SETUP_GUIDE.md](SETUP_GUIDE.md).

**Key steps from `SETUP_GUIDE.md` include:**
*   Creating or selecting a GCP Project.
*   Enabling Vertex AI, Document AI, and Cloud Storage APIs.
*   Creating a service account with appropriate roles (e.g., Vertex AI User, Document AI Editor, Storage Object Admin).
*   Downloading the service account JSON key.

## 3. Environment Variables

Before running the agent, you need to set the following environment variables. These are crucial for the agent to authenticate with Google Cloud and know where to find your resources.

*   `GOOGLE_APPLICATION_CREDENTIALS`: Path to your service account JSON key file.
    *   Example: `export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"`
*   `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID.
    *   Example: `export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"`
*   `GOOGLE_CLOUD_BUCKET`: The name of the Google Cloud Storage bucket the agent will use for storing original documents, extracted text, and metadata. You must create this bucket first.
    *   Example: `export GOOGLE_CLOUD_BUCKET="your-gcs-bucket-name"`
*   `GOOGLE_CLOUD_LOCATION`: The default Google Cloud region for Vertex AI services (e.g., "us-central1"). Some tools might allow overriding this or use specific locations for services like Document AI ("us" or "eu").
    *   Example: `export GOOGLE_CLOUD_LOCATION="us-central1"`

**Optional Environment Variables for Tool Defaults:**
Some tools might be configured to use default IDs if these are set (check tool implementation in `langchain_agent.py`):
*   `DOCAI_PROCESSOR_ID`: Default Document AI Processor ID for general document processing.
*   `VERTEX_AI_ENDPOINT_ID`: Default Vertex AI Endpoint ID for text embeddings (Note: the `get_embedding_tool` currently expects the full endpoint name like "textembedding-gecko@003" or a numerical ID for private endpoints, not just a simple name unless the tool is modified to construct the full name).

## 4. Installation of Dependencies

Clone the repository (if you haven't already) and navigate into the project directory. Then, install the required Python libraries:

```bash
pip install -r requirements.txt
```

## 5. Running the Agent

Once the setup is complete and environment variables are set, you can start the agent from the project's root directory:

```bash
python langchain_agent.py
```

You should see a welcome message and a `User>` prompt.

## 6. Interacting with the Agent

The agent uses LangChain and a Large Language Model (LLM) to understand your requests. You can type your requests in natural language. The agent will try to choose the best tool and parameters to fulfill your request. Be as specific as possible for complex tasks.

**Example Interactions:**

*   **Reading a local text file:**
    *   `User> What is the content of sample.txt?`
    *   `User> Can you read the file named 'my_notes.txt'?`
    *   (Ensure `sample.txt` or `my_notes.txt` exists in the same directory where you run the agent, or provide a full path).

*   **Extracting text from a local PDF file (using `pdfplumber` via `read_pdf_file_tool`):**
    *   `User> Read the PDF file 'report.pdf'`
    *   `User> Extract all text from './documents/annual_report.pdf'`
    *   (Ensure `report.pdf` or the specified PDF exists, or provide a full path).

*   **Fetching content from a web page:**
    *   `User> What is the content of the webpage at http://example.com?`
    *   `User> Fetch the article from https://www.someblog.com/my-article-url`

*   **Getting text embeddings (using Vertex AI):**
    *   If you want the agent to use default project and location (from env vars `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`):
        *   `User> Get an embedding for the text: "Machine learning is fascinating." using endpoint "textembedding-gecko@003"`
    *   To specify all parameters:
        *   `User> Generate an embedding for the text "Hello world" using project_id "my-gcp-project-id" location "us-central1" and endpoint_id "textembedding-gecko@003"`
    *   (The agent will use the `get_embedding_tool`. The `vertex_ai_endpoint_id` refers to the model name like "textembedding-gecko@003" or a specific deployed endpoint ID number.)

*   **Full Document Processing and Archiving (using `process_document_tool`):**
    This tool uploads a local document to GCS, processes it with Document AI, and stores extracted text and metadata in GCS.
    *   Example: `User> Process the document '/local/path/to/invoice.pdf' with mime type 'application/pdf' using GCS bucket 'my-actual-bucket' for project 'my-actual-project-id' with Document AI location 'us' and processor ID 'abcdef123456'.`
    *   If you have `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_BUCKET` environment variables set, you can use "DEFAULT" for those arguments:
        *   `User> Process document '/path/to/my_document.png' (mime type 'image/png') with Document AI processor 'your-docai-processor-id' in location 'us' using project DEFAULT and bucket DEFAULT.`
    *   The agent will use the `process_document_tool`. The more specific you are with parameters in your prompt, the better. Ensure the `local_file_path` is correct.

## 7. Tool Details and Finding IDs

The agent uses several tools internally. For some operations, especially with `process_document_tool` or `get_embedding_tool` if not using defaults, you'll need to provide specific IDs.

*   **Document AI Processor ID:**
    1.  Go to the Google Cloud Console.
    2.  Navigate to "Document AI" -> "Processors".
    3.  Select or create a processor (e.g., Form Parser, OCR Processor).
    4.  The "Processor ID" (and "Location") will be listed there.
*   **Vertex AI Endpoint ID (for embeddings):**
    *   For pre-trained Google models like Gecko, this is often the model name, e.g., `textembedding-gecko@003`.
    *   If you have deployed your own model to a Vertex AI Endpoint:
        1.  Go to the Google Cloud Console.
        2.  Navigate to "Vertex AI" -> "Online Prediction" (or "Endpoints").
        3.  Select your deployed endpoint.
        4.  The Endpoint ID is usually a long number found on the endpoint's detail page.
*   **Google Cloud Storage Bucket Name:** This is the name of the bucket you created in your GCP project for the agent to use.
*   **Project ID:** Your Google Cloud Project ID, visible in the Cloud Console.

## 8. Troubleshooting

*   **Authentication Errors (e.g., "PermissionDenied", "DefaultCredentialsError"):**
    *   Ensure `GOOGLE_APPLICATION_CREDENTIALS` is set correctly to a valid service account key file.
    *   Verify the service account has the required IAM roles (see `SETUP_GUIDE.md`).
    *   Make sure you've run `gcloud auth application-default login` if not using a service account for local testing (though a service account is recommended for agents).
*   **API Not Enabled:** If you see errors like "API ... has not been used in project ... before or it is disabled":
    *   Go to the Google Cloud Console -> APIs & Services -> Library, search for the API (e.g., "Vertex AI API", "Document AI API") and enable it.
*   **File Not Found Errors:**
    *   For local files, ensure the path provided to the agent is correct and accessible from where `langchain_agent.py` is run.
*   **Tool Not Found / Agent Not Understanding:**
    *   Try rephrasing your request. Be more specific about the action and parameters.
    *   Check the `verbose=True` output in the console (which is on by default) to see the agent's "thoughts" and why it might be failing to select a tool or what input it's trying to use.
*   **Missing GCS Bucket / Processor ID / Endpoint ID:**
    *   Ensure the GCS bucket exists and you have permissions.
    *   Double-check that the Document AI Processor ID or Vertex AI Endpoint ID is correct and the processor/endpoint is in the specified location/project.
*   **"DEFAULT" value issues:** If you use "DEFAULT" for project ID or bucket name, ensure `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_BUCKET` environment variables are correctly set.

---
If you encounter further issues, check the console output from the agent (it runs with `verbose=True` by default) for detailed error messages and the agent's reasoning process.
```

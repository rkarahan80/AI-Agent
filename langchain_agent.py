import os
from typing import Type, Optional, Dict, Any

# LangChain imports
from langchain_core.tools import BaseTool, tool # Updated import for @tool decorator
from langchain.pydantic_v1 import BaseModel, Field # For tool input schemas
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import VertexAI
from google.cloud import aiplatform # Import aiplatform for initialization

# Our existing modules
import document_ingestion
import gcp_ai_services
# Assuming DocumentMetadata is accessible from gcp_ai_services, if not, adjust import
# from gcp_ai_services import DocumentMetadata # For return type hints

# Initialize Vertex AI (important for LangChain integration)
# Needs PROJECT_ID and LOCATION. These should ideally be set by user's environment.
try:
    PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
    LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1") # Default location
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    print(f"Vertex AI initialized for project: {PROJECT_ID} in location: {LOCATION}")
except KeyError:
    print("Error: GOOGLE_CLOUD_PROJECT environment variable not set. Vertex AI LLM may not function.")
    PROJECT_ID = None # Explicitly set to None if not found
    LOCATION = None
except Exception as e:
    print(f"Error initializing Vertex AI: {e}")
    PROJECT_ID = None
    LOCATION = None

# --- Input Schemas for Tools ---

class ProcessDocumentInput(BaseModel):
    local_file_path: str = Field(description="Local path to the document file to be processed.")
    mime_type: str = Field(description="MIME type of the document (e.g., 'application/pdf', 'text/plain').")
    gcs_bucket_name: str = Field(description="Google Cloud Storage bucket name for storing files. If 'DEFAULT', uses environment variable GOOGLE_CLOUD_BUCKET.")
    doc_ai_project_id: str = Field(description="Google Cloud Project ID for Document AI. If 'DEFAULT', uses environment variable GOOGLE_CLOUD_PROJECT.")
    doc_ai_location: str = Field(description="Google Cloud location for Document AI processor (e.g., 'us', 'eu'). If 'DEFAULT', uses 'us'.")
    doc_ai_processor_id: str = Field(description="ID of the Document AI processor to use.")

class GetEmbeddingInput(BaseModel):
    text_content: str = Field(description="The text content for which to get the embedding.")
    vertex_ai_project_id: str = Field(description="Google Cloud Project ID for Vertex AI. If 'DEFAULT', uses environment variable GOOGLE_CLOUD_PROJECT.")
    vertex_ai_location: str = Field(description="Google Cloud location for Vertex AI endpoint (e.g., 'us-central1'). If 'DEFAULT', uses 'us-central1'.")
    vertex_ai_endpoint_id: str = Field(description="Vertex AI Endpoint ID for the text embedding model (e.g., 'textembedding-gecko@003').")

class ReadTextFileInput(BaseModel):
    file_path: str = Field(description="The local path to the text file.")

class ReadPdfFileInput(BaseModel):
    file_path: str = Field(description="The local path to the PDF file.")

class FetchWebPageContentInput(BaseModel):
    url: str = Field(description="The URL of the web page to fetch content from.")


# --- LangChain Tools ---

@tool(args_schema=ProcessDocumentInput)
def process_document_tool(local_file_path: str, mime_type: str, gcs_bucket_name: str, doc_ai_project_id: str, doc_ai_location: str, doc_ai_processor_id: str) -> Dict[str, Any]:
    """Processes a document using Google Cloud Document AI and stores results in GCS.
    This involves uploading the original document, calling Document AI for text extraction,
    and storing both the extracted text and metadata JSON in Google Cloud Storage."""

    current_project_id = PROJECT_ID if PROJECT_ID else os.environ.get("GOOGLE_CLOUD_PROJECT")
    current_gcs_bucket = os.environ.get("GOOGLE_CLOUD_BUCKET")

    if gcs_bucket_name == "DEFAULT":
        gcs_bucket_name = current_gcs_bucket
    if doc_ai_project_id == "DEFAULT":
        doc_ai_project_id = current_project_id
    if doc_ai_location == "DEFAULT":
        doc_ai_location = "us"

    if not all([gcs_bucket_name, doc_ai_project_id, doc_ai_location, doc_ai_processor_id]):
        return {"error": "Missing GCS bucket name, Document AI project ID, location, or processor ID. Check environment variables or provide them."}
    if not local_file_path or not mime_type:
        return {"error": "Missing local_file_path or mime_type."}

    # Assuming gcp_ai_services.orchestrate_document_processing returns a dict (from DocumentMetadata.to_dict())
    metadata_dict = gcp_ai_services.orchestrate_document_processing(
        project_id=doc_ai_project_id,
        location=doc_ai_location,
        processor_id=doc_ai_processor_id,
        local_file_path=local_file_path, # Changed from file_path to local_file_path
        mime_type=mime_type,
        bucket_name=gcs_bucket_name
    )
    if metadata_dict:
        return metadata_dict
    return {"error": "Failed to process document or no metadata returned."}

@tool(args_schema=GetEmbeddingInput)
def get_embedding_tool(text_content: str, vertex_ai_project_id: str, vertex_ai_location: str, vertex_ai_endpoint_id: str) -> Dict[str, Any]:
    """Generates a text embedding using a Vertex AI endpoint."""
    current_project_id = PROJECT_ID if PROJECT_ID else os.environ.get("GOOGLE_CLOUD_PROJECT")
    # Use existing LOCATION for Vertex if initialized, otherwise a common default
    current_vertex_location = LOCATION if LOCATION else "us-central1"


    if vertex_ai_project_id == "DEFAULT":
        vertex_ai_project_id = current_project_id
    if vertex_ai_location == "DEFAULT":
        vertex_ai_location = current_vertex_location

    if not all([vertex_ai_project_id, vertex_ai_location, vertex_ai_endpoint_id]):
         return {"error": "Missing Vertex AI project ID, location, or endpoint ID. Check environment variables or provide them."}
    if not text_content:
        return {"error": "Missing text_content for embedding."}

    embedding = gcp_ai_services.get_text_embedding_vertex_ai(
        project_id=vertex_ai_project_id,
        location=vertex_ai_location,
        endpoint_id=vertex_ai_endpoint_id,
        text_content=text_content
    )
    if embedding:
        return {"embedding_vector": embedding, "dimensionality": len(embedding)} # Changed key to "embedding_vector"
    return {"error": "Failed to get embedding."}

@tool(args_schema=ReadTextFileInput)
def read_text_file_tool(file_path: str) -> str:
    """Reads content from a local text file. Returns the content as a string or an error message."""
    try:
        return document_ingestion.read_text_file(file_path)
    except Exception as e:
        return f"Error reading text file {file_path}: {e}"

@tool(args_schema=ReadPdfFileInput)
def read_pdf_file_tool(file_path: str) -> str:
    """Extracts text content from a local PDF file using pdfplumber. Returns the extracted text or an error message."""
    try:
        # Add a check for file existence for robustness, as pdfplumber might error out.
        if not os.path.exists(file_path):
            return f"Error: PDF file not found at {file_path}"
        return document_ingestion.read_pdf_file(file_path)
    except Exception as e:
        return f"Error reading PDF file {file_path}: {e}"

@tool(args_schema=FetchWebPageContentInput)
def fetch_web_page_content_tool(url: str) -> str:
    """Fetches and extracts textual content from a given web page URL. Returns the content or an error message."""
    try:
        return document_ingestion.fetch_web_page_content(url)
    except Exception as e:
        return f"Error fetching web page content from {url}: {e}"

# --- List of Tools ---
tools = [
    process_document_tool,
    get_embedding_tool,
    read_text_file_tool,
    read_pdf_file_tool,
    fetch_web_page_content_tool
]

# --- Agent Initialization and Main Loop ---
from langchain import hub

if __name__ == '__main__':
    if not PROJECT_ID or not LOCATION:
        print("CRITICAL ERROR: GOOGLE_CLOUD_PROJECT and/or GOOGLE_CLOUD_LOCATION (for Vertex AI) are not set or Vertex AI failed to initialize. LangChain Agent cannot start.")
        print("Please ensure these environment variables are correctly set and aiplatform.init() succeeds.")
    else:
        try:
            # Using a more specific model name, ensure this model is available in your project/location
            llm = VertexAI(model_name="gemini-1.0-pro", project=PROJECT_ID, location=LOCATION)
            print(f"VertexAI LLM initialized with model gemini-1.0-pro in project {PROJECT_ID} location {LOCATION}.")
        except Exception as e:
            print(f"Error initializing VertexAI LLM: {e}. Agent will not run.")
            llm = None

        if llm:
            # Pull a standard ReAct prompt.
            try:
                prompt = hub.pull("hwchase17/react-json")
                # Ensure the prompt's input variables are compatible.
                # create_react_agent expects "input", "tools", and "agent_scratchpad".
                # The hwchase17/react-json prompt is generally suitable.
                # It might also use "tool_names", which AgentExecutor provides.
                print("Pulled ReAct prompt 'hwchase17/react-json' from Langchain Hub.")
            except Exception as e:
                print(f"Error pulling 'hwchase17/react-json' prompt from Langchain Hub: {e}. Using a basic fallback prompt.")
                # Fallback prompt (very basic, might not work as well as a tuned one)
                from langchain_core.prompts import PromptTemplate
                prompt_str = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
                prompt = PromptTemplate.from_template(prompt_str)
                print("Using basic fallback prompt template.")

            try:
                agent = create_react_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=True, # Set to False for less detailed output
                    handle_parsing_errors="Check your output and make sure it conforms!", # Provides a message on parsing error
                    max_iterations=10 # Prevents overly long loops
                )
                print("LangChain ReAct Agent initialized successfully.")
            except Exception as e:
                print(f"Error initializing LangChain agent or executor: {e}")
                agent_executor = None


            if agent_executor:
                print("\nLangChain ReAct Agent is ready. Type 'exit' to quit.")
                print("Example commands you can try (ensure dummy files like 'sample.txt' exist if needed):")
                print("  - 'What is the content of sample.txt?' (Assumes sample.txt exists)")
                print("  - 'Process the document orchestration_sample.pdf with mime type application/pdf using processor <YOUR_PROCESSOR_ID> in project DEFAULT bucket DEFAULT location DEFAULT.'")
                print("  - 'Get an embedding for the text: Hello intelligent world'")

                # Create a dummy sample.txt for testing read_text_file_tool
                if not os.path.exists("sample.txt"):
                    with open("sample.txt", "w", encoding="utf-8") as f:
                        f.write("This is a sample text file for the LangChain agent to read using its tools.")
                    print("Created 'sample.txt' for testing.")

                while True:
                    try:
                        user_input = input("User> ")
                        if user_input.lower() == 'exit':
                            break
                        if not user_input.strip():
                            continue

                        # The agent_executor will fill in 'agent_scratchpad', 'tools', 'tool_names'
                        response = agent_executor.invoke({"input": user_input})
                        print(f"Agent: {response.get('output', 'No output found in response.')}")

                    except KeyboardInterrupt:
                        print("\nExiting agent interaction loop...")
                        break
                    except Exception as e:
                        print(f"Error during agent interaction: {e}")
                        # import traceback
                        # traceback.print_exc() # Uncomment for detailed stack trace
                        # Depending on the error, you might want to break or allow continuation.
                        # For now, we continue the loop.
            else:
                print("Agent Executor not initialized. Cannot start interaction loop.")
        else:
            print("LLM not initialized. Agent cannot run.")
    print("\nLangChain agent session finished.")

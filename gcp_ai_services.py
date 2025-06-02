# gcp_ai_services.py

"""
This module provides functions to interact with Google Cloud AI services,
such as Document AI and Vertex AI.
"""

from google.cloud import documentai_v1 as documentai
from google.cloud import aiplatform
from google.cloud import storage
# from google.protobuf import struct_pb2 # May not be needed if using client library methods directly
from typing import Optional, List, Dict, Any
import os
import uuid
import datetime
import json

# --- Metadata Structure ---

class DocumentMetadata:
    def __init__(
        self,
        document_id: str,
        source_uri: Optional[str] = None,
        gcs_raw_path: Optional[str] = None,
        gcs_extracted_text_path: Optional[str] = None,
        gcs_metadata_path: Optional[str] = None,
        document_type: Optional[str] = None,
        mime_type: Optional[str] = None,
        ingestion_date: Optional[str] = None,
        processing_status: str = "pending", # e.g., pending, succeeded, failed, partial
        extracted_text_snippet: Optional[str] = None,
        error_message: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None, # Placeholder for future entity extraction
        summary: Optional[str] = None, # Placeholder for future summarization
        embedding_vector_id: Optional[str] = None # Placeholder for ID linking to an embedding vector
    ):
        self.document_id = document_id
        self.source_uri = source_uri
        self.gcs_raw_path = gcs_raw_path
        self.gcs_extracted_text_path = gcs_extracted_text_path
        self.gcs_metadata_path = gcs_metadata_path
        self.document_type = document_type
        self.mime_type = mime_type
        self.ingestion_date = ingestion_date or datetime.datetime.utcnow().isoformat() + "Z"
        self.processing_status = processing_status
        self.extracted_text_snippet = extracted_text_snippet
        self.error_message = error_message
        self.entities = entities or {}
        self.summary = summary
        self.embedding_vector_id = embedding_vector_id

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_string: str) -> 'DocumentMetadata':
        data = json.loads(json_string)
        return cls(**data)

def process_document_with_document_ai(
    project_id: str,
    location: str,
    processor_id: str,
    file_path: str,
    mime_type: str
) -> Optional[str]:
    """
    Processes a document using Google Cloud Document AI.

    This function takes a local file path, sends it to the specified
    Document AI processor, and returns the extracted text.
    Currently, this function only supports local file paths. GCS URIs
    will be supported in a future update.

    Args:
        project_id: The Google Cloud Project ID.
        location: The location of the Document AI processor (e.g., "us", "eu").
        processor_id: The ID of the Document AI processor.
        file_path: The local path to the document file (e.g., PDF, JPEG, PNG).
        mime_type: The MIME type of the document (e.g., "application/pdf", "image/jpeg").

    Returns:
        The extracted text content from the document as a string,
        or None if an error occurs or no text is extracted.
    """
    opts = {}
    if location != "us": # Document AI API has regional endpoints
        opts["api_endpoint"] = f"{location}-documentai.googleapis.com"

    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    name = client.processor_path(project_id, location, processor_id)

    try:
        # Read the file content in binary mode
        with open(file_path, "rb") as document_file:
            file_content = document_file.read()

        # Load Binary Data into Document AI RawDocument Object
        raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)

        # Configure the process request
        request = documentai.ProcessRequest(name=name, raw_document=raw_document)

        # Use the Document AI client to process the sample document
        result = client.process_document(request=request)

        # Extract and return the document text
        return result.document.text

    except FileNotFoundError:
        print(f"Error: Document file not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while processing the document with Document AI: {e}")
        return None

def get_text_embedding_vertex_ai(
    project_id: str,
    location: str,
    endpoint_id: str, # This is the DeployedIndexEndpoint ID or a public/pre-trained model endpoint
    text_content: str
) -> Optional[List[float]]:
    """
    Gets text embeddings from a Vertex AI endpoint.

    Args:
        project_id: The Google Cloud Project ID.
        location: The location of the Vertex AI endpoint (e.g., "us-central1").
        endpoint_id: The ID of the Vertex AI Endpoint.
                     For pre-trained models like 'textembedding-gecko', this is the model name.
                     For DeployedIndexEndpoints, this is the numerical ID of the endpoint.
        text_content: The text content to get embeddings for.

    Returns:
        A list of floats representing the embedding vector, or None if an error occurs.
    """
    try:
        aiplatform.init(project=project_id, location=location)

        # For public models like textembedding-gecko@001, the endpoint_id is the model name.
        # For private endpoints (DeployedIndexEndpoint), it's a numerical ID.
        # The SDK handles constructing the full endpoint path.
        # Example for a pre-trained model: "textembedding-gecko@001"
        # Example for a deployed private endpoint: "YOUR_ENDPOINT_ID_NUMBER"
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id) # Simplified endpoint creation

        instances = [{"content": text_content}]

        response = endpoint.predict(instances=instances)

        if response.predictions and len(response.predictions) > 0:
            # The structure of response.predictions[0] can be a dict or a protobuf Struct
            prediction_result = response.predictions[0]

            # Handling both dict and protobuf Struct
            if isinstance(prediction_result, dict):
                embeddings_dict = prediction_result.get('embeddings')
            # elif hasattr(prediction_result, 'pb_value'): # Check if it's a protobuf Struct
            #     embeddings_dict = dict(prediction_result.pb_value.get('embeddings', {}))
            else: # Assuming it's a protobuf Struct-like object from aiplatform.Prediction
                 embeddings_dict = getattr(prediction_result, 'embeddings', None)


            if embeddings_dict:
                embedding_values = getattr(embeddings_dict, 'values', None) # For protobuf Struct
                if embedding_values is not None:
                     return list(embedding_values) # Convert to list
                elif isinstance(embeddings_dict, dict) and 'values' in embeddings_dict: # For dict
                    return embeddings_dict['values']
                else:
                    print("Error: 'values' field not found in 'embeddings' of prediction.")
                    print(f"Embeddings structure: {embeddings_dict}")
                    return None
            else:
                print("Error: 'embeddings' field not found in prediction.")
                print(f"Prediction structure: {prediction_result}")
                return None
        else:
            print("Error: Empty or no predictions received from Vertex AI endpoint.")
            print(f"Full response: {response}")
            return None

    except Exception as e:
        print(f"An error occurred while calling Vertex AI endpoint: {e}")
        # Log the full error for more details, e.g., using logging module
        # import traceback
        # print(traceback.format_exc())
        return None

# --- Google Cloud Storage Functions ---

def upload_to_gcs(
    bucket_name: str,
    source_file_path: str,
    destination_blob_name: str
) -> Optional[str]:
    """
    Uploads a local file to Google Cloud Storage.

    Args:
        bucket_name: The name of the GCS bucket.
        source_file_path: The local path of the file to upload.
        destination_blob_name: The desired name of the object in GCS (e.g., "folder/file.txt").

    Returns:
        The GCS URI (gs://bucket_name/destination_blob_name) of the uploaded file, or None on failure.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_path)
        gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
        print(f"File {source_file_path} uploaded to {gcs_uri}.")
        return gcs_uri
    except FileNotFoundError:
        print(f"Error: Source file not found at {source_file_path}.")
        return None
    except Exception as e:
        print(f"Failed to upload {source_file_path} to gs://{bucket_name}/{destination_blob_name}: {e}")
        return None

def download_from_gcs(
    bucket_name: str,
    source_blob_name: str,
    destination_file_path: str
) -> bool:
    """
    Downloads an object from Google Cloud Storage to a local file.

    Args:
        bucket_name: The name of the GCS bucket.
        source_blob_name: The name of the object in GCS to download.
        destination_file_path: The local path where the file should be saved.

    Returns:
        True if download was successful, False otherwise.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)

        blob.download_to_filename(destination_file_path)
        print(f"Blob {source_blob_name} from bucket {bucket_name} downloaded to {destination_file_path}.")
        return True
    except Exception as e:
        print(f"Failed to download gs://{bucket_name}/{source_blob_name} to {destination_file_path}: {e}")
        return False

def store_text_in_gcs(
    bucket_name: str,
    destination_blob_name: str,
    text_content: str,
    content_type: str = "text/plain"
) -> Optional[str]:
    """
    Stores a string of text directly into a GCS object.

    Args:
        bucket_name: The name of the GCS bucket.
        destination_blob_name: The desired name of the object in GCS (e.g., "texts/my_doc.txt").
        text_content: The string content to store.
        content_type: The content type of the text (e.g., "text/plain", "application/json").

    Returns:
        The GCS URI (gs://bucket_name/destination_blob_name) of the stored text, or None on failure.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_string(text_content, content_type=content_type)
        gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
        print(f"Text content stored to {gcs_uri} with content type {content_type}.")
        return gcs_uri
    except Exception as e:
        print(f"Failed to store text to gs://{bucket_name}/{destination_blob_name}: {e}")
        return None

# --- Orchestration Function ---

def orchestrate_document_processing(
    project_id: str,
    location: str, # For Document AI Processor
    processor_id: str,
    local_file_path: str,
    mime_type: str,
    bucket_name: str,
    source_uri: Optional[str] = None # Optional original URI if not a local file initially
) -> Optional[Dict[str, Any]]:
    """
    Orchestrates the document processing pipeline:
    1. Uploads the original document to GCS.
    2. Processes the document with Document AI (from local path for now).
    3. Stores the extracted text to GCS.
    4. Populates and stores metadata to GCS.

    Args:
        project_id: Google Cloud Project ID.
        location: Location of the Document AI processor.
        processor_id: Document AI processor ID.
        local_file_path: Local path to the document file.
        mime_type: MIME type of the document.
        bucket_name: GCS bucket name for storing artifacts.
        source_uri: Optional original source URI of the document (e.g., web URL, original file path).

    Returns:
        A dictionary containing the populated metadata if successful, otherwise None or partial metadata.
    """
    doc_id = str(uuid.uuid4())
    print(f"Starting processing for new document_id: {doc_id}")

    # Define GCS paths
    file_name = os.path.basename(local_file_path)
    gcs_raw_path_suffix = f"raw_docs/{doc_id}/{file_name}"
    gcs_extracted_text_path_suffix = f"extracted_text/{doc_id}/full_text.txt"
    gcs_metadata_path_suffix = f"metadata/{doc_id}/metadata.json"

    # Initialize metadata
    metadata = DocumentMetadata(
        document_id=doc_id,
        source_uri=source_uri or local_file_path,
        mime_type=mime_type,
        document_type=mime_type.split('/')[-1] if mime_type else "unknown"
    )
    metadata.gcs_metadata_path = f"gs://{bucket_name}/{gcs_metadata_path_suffix}" # Self-reference

    # Step 1: Upload original document to GCS
    print(f"Uploading original file {local_file_path} to GCS path: {gcs_raw_path_suffix}")
    uploaded_raw_uri = upload_to_gcs(bucket_name, local_file_path, gcs_raw_path_suffix)
    if not uploaded_raw_uri:
        metadata.processing_status = "failed"
        metadata.error_message = "Failed to upload original document to GCS."
        print(metadata.error_message)
        # Attempt to store failure metadata
        store_text_in_gcs(bucket_name, gcs_metadata_path_suffix, metadata.to_json(), content_type="application/json")
        return metadata.to_dict()
    metadata.gcs_raw_path = uploaded_raw_uri
    print(f"Original document uploaded to: {uploaded_raw_uri}")

    # Step 2: Process with Document AI
    print(f"Processing document '{local_file_path}' with Document AI...")
    extracted_text = process_document_with_document_ai(
        project_id, location, processor_id, local_file_path, mime_type
    )

    if extracted_text is None:
        metadata.processing_status = "failed"
        metadata.error_message = "Document AI processing failed or returned no text."
        print(metadata.error_message)
        store_text_in_gcs(bucket_name, gcs_metadata_path_suffix, metadata.to_json(), content_type="application/json")
        return metadata.to_dict()

    if not extracted_text.strip():
        print("Warning: Document AI processing returned empty text.")
        metadata.processing_status = "succeeded_empty_text"
        metadata.extracted_text_snippet = ""
    else:
        metadata.extracted_text_snippet = (extracted_text[:250] + "...") if len(extracted_text) > 250 else extracted_text
        print(f"Document AI processing successful. Snippet: {metadata.extracted_text_snippet[:100]}...")

    # Step 3: Store extracted text to GCS
    print(f"Storing extracted text to GCS path: {gcs_extracted_text_path_suffix}")
    stored_text_uri = store_text_in_gcs(
        bucket_name, gcs_extracted_text_path_suffix, extracted_text, content_type="text/plain"
    )
    if not stored_text_uri:
        metadata.processing_status = "partial_failure" # Succeeded DocAI, but failed text storage
        metadata.error_message = (metadata.error_message + "; " if metadata.error_message else "") + "Failed to store extracted text to GCS."
        print(metadata.error_message)
        store_text_in_gcs(bucket_name, gcs_metadata_path_suffix, metadata.to_json(), content_type="application/json")
        return metadata.to_dict()
    metadata.gcs_extracted_text_path = stored_text_uri
    print(f"Extracted text stored at: {stored_text_uri}")

    # Step 4: Populate and store final metadata
    metadata.processing_status = "succeeded" if extracted_text.strip() else "succeeded_empty_text"
    print(f"Storing final metadata to GCS path: {gcs_metadata_path_suffix}")
    final_metadata_uri = store_text_in_gcs(
        bucket_name, gcs_metadata_path_suffix, metadata.to_json(), content_type="application/json"
    )
    if not final_metadata_uri:
        print(f"CRITICAL ERROR: Failed to store final metadata at {gcs_metadata_path_suffix}.")
        metadata.error_message = (metadata.error_message + "; CRITICAL: Failed to store final metadata."
                                  if metadata.error_message else "CRITICAL: Failed to store final metadata.")
        # The metadata object is returned, but its GCS persistence failed.
        return metadata.to_dict()

    print(f"Document processing complete for {doc_id}. Final metadata stored at {final_metadata_uri}")
    return metadata.to_dict()


if __name__ == '__main__':
    # --- Example Usage for process_document_with_document_ai ---
    # Note: To run this example, you need:
    # 1. A Google Cloud Project with the Document AI API enabled.
    # 2. A Document AI processor created (e.g., a Form Parser or OCR processor).
    # 3. The `GOOGLE_APPLICATION_CREDENTIALS` environment variable set to point to your
    #    service account key JSON file.
    # 4. A sample document file (e.g., a PDF or image).

    # Replace with your actual values
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") # Or hardcode: "your-gcp-project-id"
    LOCATION = "us"  # e.g., "us" or "eu" - Must match your processor's region
    PROCESSOR_ID = "" # e.g., "your-ocr-processor-id" or "your-form-parser-id"

    # Create a dummy PDF file for testing if you don't have one.
    # In a real scenario, replace "sample_document.pdf" with the actual path to your file.
    SAMPLE_FILE_PATH = "sample_document.pdf"
    SAMPLE_MIME_TYPE = "application/pdf"

    # Create a dummy file for the example if it doesn't exist
    # This is just for the example to run without manual file creation.
    # In a real use case, you would provide an actual document.
    if not os.path.exists(SAMPLE_FILE_PATH) and PROCESSOR_ID:
        try:
            with open(SAMPLE_FILE_PATH, "w") as f:
                f.write("This is a dummy PDF content placeholder for testing Document AI.")
            print(f"Created a dummy file: {SAMPLE_FILE_PATH} for the example.")
            print("Replace it with a real PDF or image file for actual Document AI processing.")
        except IOError as e:
            print(f"Could not create dummy file {SAMPLE_FILE_PATH}: {e}")

    print("\n--- Testing process_document_with_document_ai ---")
    if not PROJECT_ID or not LOCATION or not PROCESSOR_ID:
        print("Please set PROJECT_ID, LOCATION, and PROCESSOR_ID environment variables or hardcode them to run the example.")
        print("If using a hardcoded PROCESSOR_ID, ensure it's a valid ID for your project and location.")
        print("Example: PROCESSOR_ID = 'your-processor-id'")
    elif not os.path.exists(SAMPLE_FILE_PATH):
        print(f"Sample file '{SAMPLE_FILE_PATH}' not found. Please create it or provide a valid path.")
    else:
        print(f"Attempting to process document: {SAMPLE_FILE_PATH}")
        print(f"Using Project ID: {PROJECT_ID}, Location: {LOCATION}, Processor ID: {PROCESSOR_ID}")

        extracted_text = process_document_with_document_ai(
            PROJECT_ID, LOCATION, PROCESSOR_ID, SAMPLE_FILE_PATH, SAMPLE_MIME_TYPE
        )

        if extracted_text:
            print("\n--- Extracted Text (Document AI) ---")
            # print(extracted_text) # Full text can be very long
            print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
        else:
            print("\nNo text extracted or an error occurred during Document AI processing.")
            print("Check the console for error messages from the function.")
            print("Ensure your Document AI processor is set up correctly and supports the file type.")

    # To clean up the dummy file after the example (optional)
    # if os.path.exists(SAMPLE_FILE_PATH) and "dummy PDF content placeholder" in open(SAMPLE_FILE_PATH).read() and not PROCESSOR_ID:
    #     try:
    #         os.remove(SAMPLE_FILE_PATH)
    #         print(f"\nCleaned up dummy file: {SAMPLE_FILE_PATH}")
    #     except OSError as e:
    #         print(f"Error deleting dummy file {SAMPLE_FILE_PATH}: {e}")

    print("\n\n--- Testing get_text_embedding_vertex_ai ---")
    # Replace with your actual Vertex AI text embedding endpoint ID or a pre-trained model name
    # For example, for the gecko text embedding model: "textembedding-gecko@003" or "textembedding-gecko@latest"
    # Ensure the model or endpoint is available in the specified LOCATION.
    VERTEX_AI_ENDPOINT_ID = "textembedding-gecko@003" # Example: Use a pre-trained model
    VERTEX_AI_LOCATION = "us-central1" # Common location for many Vertex AI models

    SAMPLE_TEXT_FOR_EMBEDDING = "This is a sample text to get embeddings for using Vertex AI."

    # Assuming PROJECT_ID is already set from the Document AI example, or retrieved via os.getenv("GOOGLE_CLOUD_PROJECT")
    if not PROJECT_ID:
        print("GOOGLE_CLOUD_PROJECT environment variable not set. Please set it to your project ID.")
    elif not VERTEX_AI_ENDPOINT_ID:
        print("Please set VERTEX_AI_ENDPOINT_ID to your Vertex AI endpoint ID or a model name.")
    else:
        print(f"Attempting to get embedding for text: '{SAMPLE_TEXT_FOR_EMBEDDING}'")
        print(f"Using Project ID: {PROJECT_ID}, Location: {VERTEX_AI_LOCATION}, Endpoint ID/Model: {VERTEX_AI_ENDPOINT_ID}")

        embedding_vector = get_text_embedding_vertex_ai(
            PROJECT_ID, VERTEX_AI_LOCATION, VERTEX_AI_ENDPOINT_ID, SAMPLE_TEXT_FOR_EMBEDDING
        )

        if embedding_vector:
            print(f"\nObtained embedding for text: '{SAMPLE_TEXT_FOR_EMBEDDING}'")
            print(f"Embedding vector (first 10 dims): {embedding_vector[:10]}")
            print(f"Embedding dimensionality: {len(embedding_vector)}")
        else:
            print("\nFailed to get text embedding from Vertex AI.")
            print("Check the console for error messages.")
            print("Ensure your Vertex AI endpoint/model is correctly specified and deployed/available in the region.")
            print("Also, ensure the `aiplatform.init()` call uses the correct project and location for the endpoint.")

    print("\n\n--- Testing Google Cloud Storage Functions ---")
    GCS_BUCKET_NAME = os.getenv("GOOGLE_CLOUD_BUCKET") # Or hardcode: "your-gcs-bucket-name"

    if not GCS_BUCKET_NAME:
        print("Please set the GOOGLE_CLOUD_BUCKET environment variable or hardcode GCS_BUCKET_NAME to run GCS examples.")
    else:
        print(f"Using GCS Bucket: {GCS_BUCKET_NAME}")
        LOCAL_FILE_TO_UPLOAD = "gcs_sample_upload.txt"
        GCS_OBJECT_NAME_UPLOAD = f"test_uploads/{LOCAL_FILE_TO_UPLOAD}"
        LOCAL_FILE_TO_DOWNLOAD = "gcs_sample_downloaded.txt"
        GCS_OBJECT_NAME_TEXT = "test_texts/sample_text_data.txt"
        SAMPLE_TEXT_CONTENT_GCS = "This is some sample text to store directly in GCS using store_text_in_gcs."
        GCS_JSON_OBJECT_NAME = "test_json/sample_metadata.json"
        SAMPLE_JSON_CONTENT = '{"name": "Sample Document", "status": "processed", "item_count": 42}'

        # Create a dummy file to upload
        try:
            with open(LOCAL_FILE_TO_UPLOAD, "w", encoding="utf-8") as f:
                f.write("Hello Google Cloud Storage! This is a test file.")
            print(f"\nCreated dummy file for upload: {LOCAL_FILE_TO_UPLOAD}")

            gcs_uri_upload = upload_to_gcs(GCS_BUCKET_NAME, LOCAL_FILE_TO_UPLOAD, GCS_OBJECT_NAME_UPLOAD)
            if gcs_uri_upload:
                print(f"File successfully uploaded to: {gcs_uri_upload}")

                # Test download
                if download_from_gcs(GCS_BUCKET_NAME, GCS_OBJECT_NAME_UPLOAD, LOCAL_FILE_TO_DOWNLOAD):
                    print(f"File successfully downloaded to: {LOCAL_FILE_TO_DOWNLOAD}")
                    # Verify content (optional)
                    with open(LOCAL_FILE_TO_DOWNLOAD, "r", encoding="utf-8") as f_down:
                        downloaded_content = f_down.read()
                    with open(LOCAL_FILE_TO_UPLOAD, "r", encoding="utf-8") as f_up:
                        uploaded_content = f_up.read()
                    if downloaded_content == uploaded_content:
                        print("Downloaded content matches uploaded content. Verification successful.")
                    else:
                        print("Error: Downloaded content does NOT match uploaded content!")

                    try:
                        os.remove(LOCAL_FILE_TO_DOWNLOAD)
                        print(f"Cleaned up downloaded file: {LOCAL_FILE_TO_DOWNLOAD}")
                    except OSError as e:
                        print(f"Error cleaning up downloaded file {LOCAL_FILE_TO_DOWNLOAD}: {e}")
                else:
                    print(f"Failed to download {GCS_OBJECT_NAME_UPLOAD} from GCS.")
            else:
                print(f"Failed to upload {LOCAL_FILE_TO_UPLOAD} to GCS.")
        except IOError as e:
            print(f"IOError during file operations for upload/download: {e}")
        finally:
            # Clean up dummy upload file
            if os.path.exists(LOCAL_FILE_TO_UPLOAD):
                try:
                    os.remove(LOCAL_FILE_TO_UPLOAD)
                    print(f"Cleaned up local upload file: {LOCAL_FILE_TO_UPLOAD}")
                except OSError as e:
                    print(f"Error cleaning up local upload file {LOCAL_FILE_TO_UPLOAD}: {e}")

        # Test store_text_in_gcs for plain text
        print(f"\nAttempting to store plain text content in GCS at {GCS_OBJECT_NAME_TEXT}")
        gcs_uri_text = store_text_in_gcs(GCS_BUCKET_NAME, GCS_OBJECT_NAME_TEXT, SAMPLE_TEXT_CONTENT_GCS)
        if gcs_uri_text:
            print(f"Plain text content successfully stored at: {gcs_uri_text}")
        else:
            print(f"Failed to store plain text content at {GCS_OBJECT_NAME_TEXT} in GCS.")

        # Test store_text_in_gcs for JSON content
        print(f"\nAttempting to store JSON content in GCS at {GCS_JSON_OBJECT_NAME}")
        gcs_uri_json = store_text_in_gcs(
            GCS_BUCKET_NAME, GCS_JSON_OBJECT_NAME, SAMPLE_JSON_CONTENT, content_type="application/json"
        )
        if gcs_uri_json:
            print(f"JSON content successfully stored at: {gcs_uri_json}")
        else:
            print(f"Failed to store JSON content at {GCS_JSON_OBJECT_NAME} in GCS.")

        print("\n--- GCS Examples Finished ---")


    print("\n\n--- Testing Document Processing Orchestration ---")
    # Ensure PROJECT_ID, LOCATION, PROCESSOR_ID, GCS_BUCKET_NAME are set.
    # PROCESSOR_ID for orchestration should be suitable for the sample file type.

    SAMPLE_ORCHESTRATION_FILE_PATH = "orchestration_sample.txt"
    SAMPLE_ORCHESTRATION_MIME_TYPE = "text/plain"

    # Attempt to get a specific processor ID for text, or use the general one.
    PROCESSOR_ID_FOR_ORCHESTRATION = os.getenv("DOCAI_PROCESSOR_ID_TXT") or PROCESSOR_ID

    if not all([PROJECT_ID, LOCATION, PROCESSOR_ID_FOR_ORCHESTRATION, GCS_BUCKET_NAME]):
        print("Please ensure PROJECT_ID, LOCATION (DocAI), an appropriate PROCESSOR_ID (e.g., via DOCAI_PROCESSOR_ID_TXT or hardcoded),")
        print("and GCS_BUCKET_NAME are set to run the orchestration example.")
        if not PROCESSOR_ID_FOR_ORCHESTRATION and PROCESSOR_ID:
             print(f"Using general PROCESSOR_ID ('{PROCESSOR_ID}') for orchestration as DOCAI_PROCESSOR_ID_TXT is not set.")
        elif not PROCESSOR_ID_FOR_ORCHESTRATION and not PROCESSOR_ID:
             print("Neither DOCAI_PROCESSOR_ID_TXT nor a general PROCESSOR_ID is set. Cannot run orchestration.")

    else:
        # Create a dummy file for orchestration
        try:
            with open(SAMPLE_ORCHESTRATION_FILE_PATH, "w", encoding="utf-8") as f:
                f.write("This is a sample document for the orchestration test. It contains some plain text.")
            print(f"\nCreated dummy file for orchestration: {SAMPLE_ORCHESTRATION_FILE_PATH}")

            print(f"Starting orchestration with Processor ID: {PROCESSOR_ID_FOR_ORCHESTRATION}")
            orchestration_result_metadata = orchestrate_document_processing(
                project_id=PROJECT_ID,
                location=LOCATION,
                processor_id=PROCESSOR_ID_FOR_ORCHESTRATION,
                local_file_path=SAMPLE_ORCHESTRATION_FILE_PATH,
                mime_type=SAMPLE_ORCHESTRATION_MIME_TYPE,
                bucket_name=GCS_BUCKET_NAME,
                source_uri=f"local_test_file://{os.path.abspath(SAMPLE_ORCHESTRATION_FILE_PATH)}"
            )

            if orchestration_result_metadata:
                print("\n--- Orchestration Result Metadata ---")
                print(json.dumps(orchestration_result_metadata, indent=2))
                if orchestration_result_metadata.get("processing_status") == "succeeded":
                    print(f"\nOrchestration successful! Metadata stored at: {orchestration_result_metadata.get('gcs_metadata_path')}")
                else:
                    print(f"\nOrchestration finished with status: {orchestration_result_metadata.get('processing_status')}")
                    print(f"Error message: {orchestration_result_metadata.get('error_message')}")
            else:
                print("\nOrchestration failed or returned no metadata.")

        except Exception as e:
            print(f"An unexpected error occurred during the orchestration example: {e}")
            # import traceback
            # print(traceback.format_exc())
        finally:
            if os.path.exists(SAMPLE_ORCHESTRATION_FILE_PATH):
                try:
                    os.remove(SAMPLE_ORCHESTRATION_FILE_PATH)
                    print(f"Cleaned up dummy orchestration file: {SAMPLE_ORCHESTRATION_FILE_PATH}")
                except OSError as e_clean:
                    print(f"Error cleaning up orchestration file {SAMPLE_ORCHESTRATION_FILE_PATH}: {e_clean}")

    print("\n--- All Examples Finished ---")

# agent_core.py

"""
Core logic for the document processing agent.
Handles user commands and orchestrates calls to ingestion and GCP AI services.
"""

import document_ingestion
import gcp_ai_services
import json
import os # For getting environment variables

class Agent:
    def __init__(self):
        # Configuration can be loaded here if needed, e.g., from env vars or a config file
        self.default_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.default_gcs_bucket = os.getenv("GOOGLE_CLOUD_BUCKET")
        # Default locations can also be set
        self.default_docai_location = "us"
        self.default_vertex_location = "us-central1"
        pass

    def print_help(self):
        print("\nAvailable commands:")
        print("  PROCESS_DOCUMENT <local_file_path> <mime_type> <gcs_bucket_name> <project_id> <doc_ai_location> <doc_ai_processor_id>")
        print("    -> Orchestrates document processing and stores artifacts in GCS.")
        print("  GET_EMBEDDING <text_to_embed> <project_id> <vertex_ai_location> <vertex_ai_endpoint_id>")
        print("    -> Gets text embedding from Vertex AI.")
        print("  INGEST_TEXT <file_path>")
        print("    -> Reads and prints content from a local text file.")
        print("  INGEST_PDF <file_path>")
        print("    -> Extracts and prints text from a local PDF file (requires pdfplumber).")
        print("  INGEST_WEB <url>")
        print("    -> Fetches and prints main textual content from a web page.")
        print("  HELP")
        print("    -> Shows this help message.")
        print("  EXIT")
        print("    -> Exits the agent.\n")
        print("Ensure required environment variables like GOOGLE_APPLICATION_CREDENTIALS are set for GCP commands.")
        print("PROJECT_ID and GCS_BUCKET_NAME can often be inferred from GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_BUCKET env vars if not provided.")


    def handle_command(self, command_str: str) -> bool:
        """
        Parses and executes a given command string.

        Args:
            command_str: The command string input by the user.

        Returns:
            True if the agent should continue running, False if it should exit.
        """
        parts = command_str.strip().split()
        if not parts:
            return True # Continue on empty command

        command = parts[0].upper()
        args = parts[1:]

        try:
            if command == "PROCESS_DOCUMENT":
                if len(args) < 6:
                    print("Error: PROCESS_DOCUMENT requires <local_file_path> <mime_type> <gcs_bucket_name> <project_id> <doc_ai_location> <doc_ai_processor_id>")
                    return True
                local_file_path, mime_type, gcs_bucket_name, project_id, location, processor_id = args[0], args[1], args[2], args[3], args[4], args[5]

                # Use defaults if available and args are "DEFAULT" or similar (optional enhancement)
                gcs_bucket_name = gcs_bucket_name if gcs_bucket_name != "DEFAULT" else self.default_gcs_bucket
                project_id = project_id if project_id != "DEFAULT" else self.default_project_id
                location = location if location != "DEFAULT" else self.default_docai_location

                if not all([local_file_path, mime_type, gcs_bucket_name, project_id, location, processor_id]):
                    print("Error: Missing one or more required arguments for PROCESS_DOCUMENT even after defaults.")
                    print(f"Values: path={local_file_path}, mime={mime_type}, bucket={gcs_bucket_name}, proj={project_id}, loc={location}, proc_id={processor_id}")
                    return True

                print(f"Processing document: {local_file_path}...")
                metadata = gcp_ai_services.orchestrate_document_processing(
                    project_id=project_id,
                    location=location,
                    processor_id=processor_id,
                    local_file_path=local_file_path,
                    mime_type=mime_type,
                    bucket_name=gcs_bucket_name
                )
                if metadata:
                    print("Document processing finished. Metadata:")
                    print(json.dumps(metadata, indent=2))
                else:
                    print("Document processing failed or returned no metadata.")

            elif command == "GET_EMBEDDING":
                if len(args) < 4:
                    print("Error: GET_EMBEDDING requires <text_to_embed> <project_id> <vertex_ai_location> <vertex_ai_endpoint_id>")
                    return True
                text_to_embed = args[0]
                project_id, location, endpoint_id = args[1], args[2], args[3]

                project_id = project_id if project_id != "DEFAULT" else self.default_project_id
                location = location if location != "DEFAULT" else self.default_vertex_location

                if not all([text_to_embed, project_id, location, endpoint_id]):
                     print("Error: Missing one or more required arguments for GET_EMBEDDING even after defaults.")
                     return True

                print(f"Getting embedding for text: '{text_to_embed[:50]}...'")
                embedding = gcp_ai_services.get_text_embedding_vertex_ai(
                    project_id=project_id,
                    location=location,
                    endpoint_id=endpoint_id,
                    text_content=text_to_embed
                )
                if embedding:
                    print(f"Obtained embedding (first 10 dims): {embedding[:10]}")
                    print(f"Dimensionality: {len(embedding)}")
                else:
                    print("Failed to get text embedding.")

            elif command == "INGEST_TEXT":
                if len(args) < 1:
                    print("Error: INGEST_TEXT requires <file_path>")
                    return True
                content = document_ingestion.read_text_file(args[0])
                if content:
                    print("Extracted text content (first 500 chars):")
                    print(content[:500] + ("..." if len(content) > 500 else ""))
                else:
                    print(f"No content extracted or file not found: {args[0]}")

            elif command == "INGEST_PDF":
                if len(args) < 1:
                    print("Error: INGEST_PDF requires <file_path>")
                    return True
                # Create a dummy PDF for testing if it doesn't exist, as pdfplumber needs a real file
                if not os.path.exists(args[0]):
                     print(f"Warning: PDF file {args[0]} not found. PDF ingestion will likely fail or process nothing.")
                content = document_ingestion.read_pdf_file(args[0])
                if content:
                    print("Extracted PDF content (first 500 chars):")
                    print(content[:500] + ("..." if len(content) > 500 else ""))
                else:
                    print(f"No content extracted from PDF or error: {args[0]}")

            elif command == "INGEST_WEB":
                if len(args) < 1:
                    print("Error: INGEST_WEB requires <url>")
                    return True
                content = document_ingestion.fetch_web_page_content(args[0])
                if content:
                    print("Fetched web content (first 500 chars):")
                    print(content[:500] + ("..." if len(content) > 500 else ""))
                else:
                    print(f"No content fetched from URL or error: {args[0]}")

            elif command == "HELP":
                self.print_help()

            elif command == "EXIT":
                print("Exiting agent...")
                return False # Signal to exit

            else:
                print(f"Unknown command: {command}. Type HELP for available commands.")

        except Exception as e:
            print(f"An error occurred while executing command '{command}': {e}")
            # import traceback
            # traceback.print_exc() # For more detailed debugging

        return True # Continue by default


if __name__ == "__main__":
    print("Document Processing Agent Initialized.")
    print("Type 'HELP' for a list of commands.")
    print("Ensure GOOGLE_APPLICATION_CREDENTIALS is set in your environment for GCP operations.")
    print("You can use 'DEFAULT' for project_id, gcs_bucket_name, and location arguments")
    print("if GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_BUCKET are set in your environment.")

    agent = Agent()

    # Create dummy files for easy testing of ingestion commands
    if not os.path.exists("agent_sample.txt"):
        with open("agent_sample.txt", "w") as f:
            f.write("This is a sample text file for the agent to ingest.")
    # Note: A dummy PDF is harder to create meaningfully on the fly without a library.
    # The INGEST_PDF command will warn if the file is missing.

    running = True
    while running:
        try:
            user_input = input("Agent> ")
            if user_input.strip(): # Avoid processing empty inputs immediately
                 running = agent.handle_command(user_input)
        except KeyboardInterrupt:
            print("\nExiting agent due to KeyboardInterrupt...")
            running = False
        except EOFError: # Handle EOF (e.g. if input is piped)
            print("\nExiting agent due to EOF...")
            running = False

    print("Agent terminated.")

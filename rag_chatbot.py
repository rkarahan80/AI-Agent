import os

# --- OpenAI API Key Setup ---
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    if 'OPENAI_API_KEY' in os.environ:
        OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    else:
        print("Error: OPENAI_API_KEY environment variable not found.")
        print("Please set it before running the script. e.g., export OPENAI_API_KEY='your_actual_key_here'")
        # This print is for the user running the script, the subtask should not exit itself.
        # If the key is critical for the subtask to complete its specific file operations,
        # then the subtask might report failure if it can't proceed.
        # For now, the goal is to write the file. The key check is for runtime.

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Removed direct OpenAIEmbeddings and OpenAI imports
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma, FAISS
from ai_models import AIModel, OpenAIModel, JulesModel, MCPModel # Import AIModel, JulesModel, and MCPModel

def load_documents(filepath="training_data.txt"):
    loader = TextLoader(filepath, encoding='utf-8') # Specify encoding
    all_docs = loader.load()

    greeting_message = "Welcome to the RAG Chatbot!" # Default greeting
    processed_documents = []

    if all_docs:
        first_doc_content = all_docs[0].page_content
        if first_doc_content.startswith("GREETING: "):
            greeting_message = first_doc_content[len("GREETING: "):].strip()
            # If there are more documents, they are the actual documents
            if len(all_docs) > 1:
                processed_documents = all_docs[1:]
            # If only the greeting line existed, documents list remains empty
            # This is handled by returning processed_documents which would be []
        else:
            # No greeting line, so all loaded documents are actual documents
            processed_documents = all_docs

    # if all_docs was empty, processed_documents remains empty, greeting is default.
    # This is fine.

    return greeting_message, processed_documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    texts = text_splitter.split_documents(documents)
    return texts

def setup_rag_pipeline(texts, ai_model: AIModel): # Accept AIModel base class instance
    print("Initializing embeddings model...")
    try:
        embeddings = ai_model.get_embeddings() # Use ai_model
    except NotImplementedError:
        print(f"Error: The get_embeddings method for the selected AI model ({type(ai_model).__name__}) is not implemented.")
        raise
    except Exception as e:
        print(f"Error getting embeddings from {type(ai_model).__name__}: {e}")
        raise

    if embeddings is None and type(ai_model).__name__ not in ["JulesModel", "MCPModel"]: # Allow placeholders to return None
        # Added a check for JulesModel/MCPModel to avoid erroring out if it's the one returning None as per its placeholder nature.
        # However, a fully functional system would expect a valid embeddings object or an explicit error.
        print(f"Error: Embeddings object is None for {type(ai_model).__name__}, and it's not an expected placeholder behavior.")
        raise ValueError("Embeddings object cannot be None for a functional RAG pipeline.")

    print("Initializing vector store...")
    vectorstore = None
    try:
        print("Attempting to use Chroma vector store...")
        vectorstore = Chroma.from_documents(texts, embeddings)
        print("Using Chroma vector store.")
    except Exception as e_chroma:
        print(f"Chroma initialization failed: {e_chroma}")
        print("Attempting to use FAISS vector store as fallback...")
        try:
            vectorstore = FAISS.from_documents(texts, embeddings) # embeddings is now the OpenAIEmbeddings instance
            print("Using FAISS vector store.")
        except Exception as e_faiss:
            print(f"FAISS initialization failed: {e_faiss}")
            print("Error: Could not initialize any suitable vector store. Check dependencies and environment.")
            raise RuntimeError("Failed to initialize vector store") from e_faiss

    if not vectorstore:
        # This case should ideally be caught by the exceptions above, but as a safeguard:
        raise RuntimeError("Vector store initialization failed.")

    print("Initializing LLM...")
    try:
        llm = ai_model.generate_response() # Use ai_model
    except NotImplementedError:
        print(f"Error: The generate_response method for the selected AI model ({type(ai_model).__name__}) is not implemented.")
        raise
    except Exception as e:
        print(f"Error getting LLM from {type(ai_model).__name__}: {e}")
        raise

    if llm is None and type(ai_model).__name__ not in ["JulesModel", "MCPModel"]: # Similar check for LLM for placeholders
        print(f"Error: LLM object is None for {type(ai_model).__name__}, and it's not an expected placeholder behavior.")
        raise ValueError("LLM object cannot be None for a functional RAG pipeline.")

    # If embeddings or llm is None here, it implies a placeholder model (JulesModel or MCPModel) is selected
    # and returned None (as per current placeholder design, though they raise NotImplementedError).
    # The RAG chain setup will likely fail if these are None. This is acceptable for this subtask.

    print("Creating retriever...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("Creating RAG chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )
    print("RAG pipeline setup complete.")
    return qa_chain

def main():
    print("Starting RAG chatbot application...")

    # Determine AI model provider
    ai_provider = os.getenv('AI_MODEL_PROVIDER', 'OPENAI').upper() # Default to OPENAI
    ai_model_instance: AIModel # Type hint for clarity

    if ai_provider == 'OPENAI':
        current_openai_api_key = OPENAI_API_KEY # Use the globally checked key
        if not current_openai_api_key:
            print("CRITICAL: OpenAI API Key (OPENAI_API_KEY) not found for OpenAI model. The script cannot run.")
            print("Please ensure OPENAI_API_KEY is set as an environment variable.")
            return
        try:
            ai_model_instance = OpenAIModel(api_key=current_openai_api_key)
            print("Using OpenAI model.")
        except Exception as e:
            print(f"Error instantiating OpenAIModel: {e}")
            return
    elif ai_provider == 'JULES':
        # Assuming Jules might use a different API key or no key for this example
        jules_api_key = os.getenv('JULES_API_KEY', 'dummy_jules_key') # Example for Jules API key
        try:
            ai_model_instance = JulesModel(api_key=jules_api_key)
            print("Using Jules model (placeholder).")
        except Exception as e:
            print(f"Error instantiating JulesModel: {e}")
            return
    elif ai_provider == 'MCP':
        # Assuming MCP might use a different API key or no key for this example
        mcp_api_key = os.getenv('MCP_API_KEY', 'dummy_mcp_key') # Example for MCP API key
        try:
            ai_model_instance = MCPModel(api_key=mcp_api_key)
            print("Using MCP model (placeholder).")
        except Exception as e:
            print(f"Error instantiating MCPModel: {e}")
            return
    else:
        print(f"Error: Unknown AI_MODEL_PROVIDER '{ai_provider}'. Please use 'OPENAI', 'JULES', or 'MCP'.")
        return

    try:
        print("Loading documents...")
        greeting, documents = load_documents() # Updated call
        if not documents: # Check documents specifically
            print("No documents found or loaded from training_data.txt (excluding potential greeting line). The file might be empty or effectively empty.")
            # Depending on desired behavior, you might still want to run the chatbot if only a greeting exists.
            # For now, if documents list is empty, we exit.
            return
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    try:
        print("Splitting documents into manageable chunks...")
        texts = split_documents(documents)
        if not texts:
            print("Document splitting resulted in no text chunks. Check document content and splitter settings.")
            return
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return

    try:
        print("Setting up the RAG pipeline. This may take a moment for embedding generation...")
        # Pass the chosen ai_model_instance to the RAG pipeline setup
        qa_chain = setup_rag_pipeline(texts, ai_model_instance)
    except NotImplementedError:
        print("Cannot proceed with RAG pipeline due to unimplemented methods in the selected AI model.")
        print(f"If using {type(ai_model_instance).__name__}, this is expected if it's a placeholder.")
        return # Exit if core model functionalities are not implemented
    except RuntimeError as e:
        print(f"Failed to set up RAG pipeline: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during RAG pipeline setup: {e}")
        return

    print(f"\n{greeting}") # Use the greeting variable
    print("The chatbot is ready. Type your question and press Enter.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            query = input("Query: ")
            if query.lower() in ['exit', 'quit']:
                print("Exiting chatbot. Goodbye!")
                break
            if not query.strip():
                print("Please enter a question.")
                continue

            print("Processing your query...")
            result = qa_chain.invoke({"query": query})

            print("\nAnswer:")
            print(result["result"])

            if result.get("source_documents"):
                print("\nSource Documents used:")
                for i, doc in enumerate(result["source_documents"]):
                    source_info = f"Source {i+1}: "
                    if 'source' in doc.metadata:
                        source_info += f"File: {doc.metadata['source']}"
                    if 'start_index' in doc.metadata:
                         source_info += f", Start Index: {doc.metadata['start_index']}"
                    # print(f"    Content Snippet: {doc.page_content[:100]}...") # Optional: content snippet
                    print(source_info)
            print("-" * 30)

        except EOFError:
            print("\nExiting chatbot due to end of input.")
            break
        except KeyboardInterrupt:
            print("\nExiting chatbot due to user interruption.")
            break
        except Exception as e:
            print(f"An error occurred while processing your query: {e}")
            # Decide whether to break or continue
            # break

if __name__ == "__main__":
    # The OPENAI_API_KEY check here is a final guard for direct script execution.
    # The main() function also checks, but this ensures the environment is sane even before calling main().
    if not OPENAI_API_KEY:
         # This message is for the person running the script.
        print("CRITICAL (module level): OpenAI API Key not found in environment. The script cannot run.")
        print("Please ensure OPENAI_API_KEY is set as an environment variable.")
    else:
        main()

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
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma, FAISS

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

def setup_rag_pipeline(texts, openai_api_key):
    print("Initializing embeddings model...")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

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
            vectorstore = FAISS.from_documents(texts, embeddings)
            print("Using FAISS vector store.")
        except Exception as e_faiss:
            print(f"FAISS initialization failed: {e_faiss}")
            print("Error: Could not initialize any suitable vector store. Check dependencies and environment.")
            raise RuntimeError("Failed to initialize vector store") from e_faiss

    if not vectorstore:
        # This case should ideally be caught by the exceptions above, but as a safeguard:
        raise RuntimeError("Vector store initialization failed.")

    print("Initializing LLM...")
    llm = OpenAI(openai_api_key=openai_api_key)

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

    # Fetch API key at the start of main, to ensure it's available for this session.
    # The global OPENAI_API_KEY is checked at module load, but good to have it explicitly for main's logic.
    current_openai_api_key = os.getenv('OPENAI_API_KEY')
    if not current_openai_api_key and 'OPENAI_API_KEY' in os.environ:
        current_openai_api_key = os.environ['OPENAI_API_KEY']

    if not current_openai_api_key:
        print("CRITICAL: OpenAI API Key not found in environment when starting main(). The script cannot run.")
        print("Please ensure OPENAI_API_KEY is set as an environment variable.")
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
        qa_chain = setup_rag_pipeline(texts, current_openai_api_key)
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

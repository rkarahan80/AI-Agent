# gui_chatbot_streamlit.py
import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma, FAISS

# --- OpenAI API Key Setup ---
# Relies on environment variable OPENAI_API_KEY.
# The .env loading at the end is a local convenience for development.
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# --- RAG Functions (copied and adapted from rag_chatbot.py) ---

def load_documents(filepath="training_data.txt"):
    loader = TextLoader(filepath, encoding='utf-8')
    all_docs = loader.load()
    greeting_message = "Welcome to the RAG Chatbot!" # Default greeting
    processed_documents = []
    if all_docs:
        first_doc_content = all_docs[0].page_content
        if first_doc_content.startswith("GREETING: "):
            greeting_message = first_doc_content[len("GREETING: "):].strip()
            if len(all_docs) > 1:
                processed_documents = all_docs[1:]
        else:
            processed_documents = all_docs
    return greeting_message, processed_documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    texts = text_splitter.split_documents(documents)
    return texts

def setup_rag_pipeline(texts, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = None
    try:
        # Try Chroma first
        vectorstore = Chroma.from_documents(texts, embeddings)
    except Exception as e_chroma:
        # Fallback to FAISS if Chroma fails
        # print(f"Chroma initialization failed: {e_chroma}. Falling back to FAISS.") # Optional: log to console
        try:
            vectorstore = FAISS.from_documents(texts, embeddings)
        except Exception as e_faiss:
            raise RuntimeError(f"Failed to initialize vector store. Chroma error: {e_chroma}, FAISS error: {e_faiss}") from e_faiss
    if not vectorstore:
        raise RuntimeError("Vector store initialization failed after all attempts.")

    llm = OpenAI(openai_api_key=openai_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False # Set to False for cleaner Streamlit console
    )
    return qa_chain

# --- Streamlit App Logic ---

@st.cache_resource
def initialize_rag_components_cached():
    global OPENAI_API_KEY

    if not OPENAI_API_KEY:
        env_api_key = os.getenv('OPENAI_API_KEY')
        if env_api_key:
            OPENAI_API_KEY = env_api_key
        # elif "OPENAI_API_KEY" in st.secrets: # Uncomment if using Streamlit secrets
        #     OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        else:
            return None, "Welcome! Please configure the OpenAI API Key.", "CRITICAL: OPENAI_API_KEY not found. The chatbot cannot operate."

    _default_greeting = "Welcome to the RAG Chatbot!"
    try:
        _greeting, _documents = load_documents()

        if not _documents:
            return None, _greeting, "Error: No documents found in training_data.txt (excluding any greeting line). The chatbot cannot answer questions without data."

        _texts = split_documents(_documents)
        if not _texts:
            return None, _greeting, "Error: Documents were found but could not be split into processable chunks. The RAG chain cannot be built."

        _qa_chain = setup_rag_pipeline(_texts, OPENAI_API_KEY)

        return _qa_chain, _greeting, None # Success

    except FileNotFoundError:
        return None, _default_greeting, "Error: The 'training_data.txt' file was not found. Please create it in the root directory."
    except Exception as e:
        # Log the full error to console for Streamlit debugging
        # print(f"Full error during RAG initialization: {e}", flush=True)
        return None, _default_greeting, f"An unexpected error occurred during RAG component initialization: {str(e)}"

def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("📄 RAG Powered Chatbot")

    qa_chain_instance, loaded_greeting, error_msg = initialize_rag_components_cached()

    # Initialize chat history with a greeting, regardless of RAG status initially
    if "messages" not in st.session_state:
        # Use loaded_greeting if available and no critical API key error, else a generic welcome.
        # The error_msg for API key is specific.
        initial_bot_message = loaded_greeting
        if error_msg and "CRITICAL: OPENAI_API_KEY not found" in error_msg:
             initial_bot_message = "Welcome! Chatbot is currently not operational due to API key configuration issue."
        elif not loaded_greeting : # Fallback if loaded_greeting is None for some other reason
            initial_bot_message = "Welcome to the RAG Chatbot!"
        st.session_state.messages = [{"role": "assistant", "content": initial_bot_message}]

    # Display error message if RAG initialization failed critically (other than just no docs for a custom greeting)
    if error_msg:
        if not (qa_chain_instance is None and loaded_greeting and "No documents found" in error_msg) and \
           not (qa_chain_instance is None and loaded_greeting and "could not be split" in error_msg) :
             # For critical errors (API key, file not found, unexpected exceptions), show error prominently.
            st.error(error_msg)
        # If it's a non-critical error (like no docs but greeting is fine), the greeting is already set.
        # We might allow interaction if qa_chain_instance is None, but it will just say it's not operational.

    # Display chat messages from history

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            full_response_text = ""
            if not qa_chain_instance:
                # This condition means RAG setup failed critically (e.g. API key, file not found after initial greeting)
                # or non-critically (e.g. no docs, docs not splittable, but greeting was loaded)
                if error_msg and "CRITICAL" in error_msg: # Critical error already displayed
                     full_response_text = "Chatbot is not operational due to a critical configuration error."
                elif error_msg : # Non-critical error (like no docs)
                    full_response_text = f"Cannot process query: {error_msg}"
                else: # Should ideally not be reached if error_msg logic is complete
                    full_response_text = "Chatbot is not initialized. Cannot process query."
                message_placeholder.error(full_response_text)
            else:
                try:
                    response = qa_chain_instance.invoke({"query": prompt})
                    answer = response.get("result", "Sorry, I could not find an answer based on the documents.")
                    full_response_text = answer

                    if response.get("source_documents"):
                        full_response_text += "\n\n**Sources:**"
                        # Display sources in expanders below the main answer
                        for i, doc in enumerate(response["source_documents"]):
                            source_info = f"Source {i+1}: "
                            if 'source' in doc.metadata:
                                source_info += f"File: {os.path.basename(doc.metadata['source'])}"
                            if 'start_index' in doc.metadata:
                                 source_info += f", Start Index: {doc.metadata['start_index']}"

                            # Add to the main message_placeholder for now for simplicity
                            # In a more complex UI, you might render these separately
                            # For now, append to the text being built for the single markdown display
                            full_response_text += f"\n\n*{source_info}*\n>{doc.page_content[:200]}..."

                    message_placeholder.markdown(full_response_text)

                except Exception as e:
                    # print(f"Error during RAG chain invocation: {e}", flush=True) # Log to console
                    full_response_text = f"Error processing your query: {str(e)}"
                    message_placeholder.error(full_response_text)

            st.session_state.messages.append({"role": "assistant", "content": full_response_text})

if __name__ == '__main__':
    # Initial attempt to load OPENAI_API_KEY if not already set by environment
    if not OPENAI_API_KEY:
        if 'OPENAI_API_KEY' in os.environ:
             OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
        else:
            env_path = os.path.join(os.getcwd(), ".env")
            if os.path.exists(env_path):
                try:
                    with open(env_path, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#'):
                                key_val = line.strip().split('=', 1)
                                if len(key_val) == 2:
                                    key, value = key_val
                                    if key == 'OPENAI_API_KEY':
                                        OPENAI_API_KEY = value.strip().strip('"').strip("'")
                                        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
                                        # print(f"Loaded OPENAI_API_KEY from {env_path}") # Optional: for debugging
                                        break
                except Exception as e:
                    pass # Silently ignore .env loading errors, rely on cached_init for user feedback

    main()

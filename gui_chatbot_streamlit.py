# gui_chatbot_streamlit.py
import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI # OpenAI for LLM, OpenAIEmbeddings for embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma, FAISS

# --- API Key Setup ---
# API keys are resolved dynamically in initialize_rag_components_cached by checking environment variables.
# The .env loading in if __name__ == '__main__' is a local convenience.
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # Removed global static initialization

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

# Updated signature to accept api_key and selected_model_type
def setup_rag_pipeline(texts, api_key: str, selected_model_type: str):
    # LLM instantiation will be refactored in the NEXT step.
    # For now, ensure 'api_key' is used for OpenAI embeddings and LLM.
    # The 'selected_model_type' will determine which LLM class to use in the future.

    # Use OpenAIEmbeddings for now, potentially pass specific key if OpenAI model selected
    # This part might need adjustment depending on how embedding keys are handled for other models
    # For now, assume OpenAI embeddings are used, and they use the provided api_key IF it's an OpenAI model,
    # otherwise, they might try to use OPENAI_API_KEY from env if that's how they default.
    # A cleaner way would be to also select embedding model based on selected_model_type.
    llm = None
    # Determine which embedding to use. For now, default to OpenAIEmbeddings.
    # This requires OPENAI_API_KEY to be set in the environment,
    # independently of the selected chat model's API key if that model is not OpenAI.
    openai_api_key_for_embeddings = os.environ.get("OPENAI_API_KEY")
    # if not openai_api_key_for_embeddings: # Optional: raise error if embeddings must work
        # raise ValueError("CRITICAL: OPENAI_API_KEY must be set in environment for embeddings to function.")

    # Use OpenAIEmbeddings. It will use openai_api_key_for_embeddings.
    # If openai_api_key_for_embeddings is None, it might still work if Langchain/OpenAI library
    # has another way to find a key (e.g. internal config), or it will fail here.
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key_for_embeddings)

    if selected_model_type == "OpenAI":
        llm = OpenAI(openai_api_key=api_key, temperature=0) # api_key is OPENAI_API_KEY for this branch
    elif selected_model_type == "Gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0)
    elif selected_model_type == "Claude":
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", anthropic_api_key=api_key, temperature=0)
    elif selected_model_type == "Deepseek":
        llm = ChatDeepSeek(model="deepseek-chat", deepseek_api_key=api_key, temperature=0)
    else:
        raise ValueError(f"Unsupported model type: {selected_model_type}")

    # Vectorstore and Retriever setup
    vectorstore = None
    try:
        vectorstore = Chroma.from_documents(texts, embeddings)
    except Exception: # Broad exception for now, can be more specific
        try:
            vectorstore = FAISS.from_documents(texts, embeddings)
        except Exception as e_faiss:
            raise RuntimeError(f"Failed to initialize any vector store: {e_faiss}")

    if not vectorstore:
        raise RuntimeError("Vector store initialization failed after all attempts.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )
    return qa_chain

# --- Streamlit App Logic ---

@st.cache_resource(show_spinner=False)
def initialize_rag_components_cached(selected_model_type: str):
    # Determine the required API key environment variable name and the key itself
    api_key = None
    api_key_name = ""
    model_display_name = ""

    if selected_model_type == "OpenAI":
        api_key_name = "OPENAI_API_KEY"
        model_display_name = "OpenAI"
    elif selected_model_type == "Gemini":
        api_key_name = "GOOGLE_API_KEY"
        model_display_name = "Google Gemini"
    elif selected_model_type == "Claude":
        api_key_name = "ANTHROPIC_API_KEY"
        model_display_name = "Anthropic Claude"
    elif selected_model_type == "Deepseek":
        api_key_name = "DEEPSEEK_API_KEY"
        model_display_name = "Deepseek"
    else:
        return None, "Welcome!", f"Error: Unknown model type '{selected_model_type}' selected."

    api_key = os.environ.get(api_key_name)
    if not api_key:
        # Try to load from .env if not found in direct environment (for local dev)
        # This is a simplified .env load, specific to this function's context
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key_val_env = line.strip().split('=', 1)
                        if len(key_val_env) == 2 and key_val_env[0] == api_key_name:
                            api_key = key_val_env[1].strip().strip('"').strip("'")
                            break # Found the key
        if not api_key: # Still not found after .env check
            return None, f"Welcome! API Key for {model_display_name} not configured.", f"CRITICAL: {api_key_name} not found in environment variables or .env file. The chatbot cannot operate with {model_display_name}."

    _default_greeting = f"Welcome to the RAG Chatbot (using {model_display_name})!"
    try:
        # Pass the fetched api_key and selected_model_type to setup_rag_pipeline
        _greeting, _documents = load_documents()

        if not _documents:
            return None, _greeting, f"Error: No documents found for {model_display_name}. Chatbot cannot answer."

        _texts = split_documents(_documents)
        if not _texts:
            return None, _greeting, f"Error: Documents found but could not be split for {model_display_name}."

        _qa_chain = setup_rag_pipeline(_texts, api_key, selected_model_type)

        return _qa_chain, _greeting, None # Success

    except FileNotFoundError:
        return None, _default_greeting, "Error: The 'training_data.txt' file was not found."
    except Exception as e:
        # print(f"Full error during RAG initialization with {model_display_name}: {e}", flush=True)
        return None, _default_greeting, f"An unexpected error occurred during RAG component initialization for {model_display_name}: {str(e)}"

def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("📄 RAG Powered Chatbot")

    # Model Selection UI in the sidebar
    st.sidebar.title("⚙️ Configuration")
    model_options = ["OpenAI", "Gemini", "Claude", "Deepseek"]
    selected_model = st.sidebar.selectbox(
        "Choose LLM Provider:",
        options=model_options,
        index=0 # Default to OpenAI
    )
    st.sidebar.info(f"Selected Model: **{selected_model}**")


    # Pass the user-selected model to the initialization function
    # The @st.cache_resource decorator on initialize_rag_components_cached
    # will ensure it re-runs and re-caches if selected_model changes.
    qa_chain_instance, loaded_greeting, error_msg = initialize_rag_components_cached(selected_model)

    # Initialize or reset chat history if model changes or first run
    if "messages" not in st.session_state or st.session_state.get("current_model") != selected_model:
        st.session_state.current_model = selected_model # Track current model for chat history reset
        initial_bot_message = loaded_greeting
        if error_msg and "CRITICAL" in error_msg : # Critical API key error for selected model
            initial_bot_message = f"Welcome! Chatbot with {selected_model} is currently not operational. Reason: {error_msg.split('CRITICAL:')[1].strip() if 'CRITICAL:' in error_msg else error_msg}"
        elif not loaded_greeting :
            initial_bot_message = f"Welcome to the RAG Chatbot with {selected_model}!"
        st.session_state.messages = [{"role": "assistant", "content": initial_bot_message}]

    if error_msg:
        # Display error if it's critical or if it's a document-related error where no custom greeting explained it.
        # This avoids showing redundant errors if a custom greeting already explains the situation (e.g., no docs).
        is_critical_key_error = "CRITICAL:" in error_msg
        is_doc_error_without_custom_greeting = ("No documents found" in error_msg or "could not be split" in error_msg) and (loaded_greeting is None or "Welcome to the RAG Chatbot!" in loaded_greeting)
        is_file_not_found_error = "training_data.txt' file was not found" in error_msg

        if is_critical_key_error or is_doc_error_without_custom_greeting or is_file_not_found_error:
            st.error(error_msg)
        # Non-critical document errors might already be part of a custom greeting, so no separate st.error needed.
        # Or, if a qa_chain_instance is None but a greeting exists, the user will be informed upon query.

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
    # The initial API key loading here is mostly for local dev convenience if .env exists.
    # The main logic for key handling is within initialize_rag_components_cached.
    # We don't need to set the global OPENAI_API_KEY here anymore as it's model-dependent.

    # Attempt to load .env file for any relevant keys if present for local testing
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key_val = line.strip().split('=', 1)
                        if len(key_val) == 2:
                            key, value = key_val
                            # Set them as environment variables so os.environ.get() can pick them up later
                            os.environ[key] = value.strip().strip('"').strip("'")
            # print("Loaded variables from .env file into environment.") # Optional debug
        except Exception as e:
            # print(f"Note: Error loading .env file: {e}") # Optional debug
            pass # Silently ignore .env loading errors

    main()

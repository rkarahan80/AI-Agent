# gui_chatbot_streamlit.py
import streamlit as st
import os
import time
import uuid
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

# --- Chat ID Generation ---
def generate_chat_id():
    return str(uuid.uuid4())

# --- Session State Initialization ---
def initialize_global_state():
    """Initializes global session state variables if they don't exist."""
    if "archived_chats" not in st.session_state:
        st.session_state.archived_chats = [] # List of {"id": archive_id, "title": title, "messages": messages_list, "original_chat_id": original_chat_id, "model_used": model_name}
    if "model_options" not in st.session_state:
        st.session_state.model_options = ["OpenAI", "Gemini", "Claude", "Deepseek"]
    # selected_model will be initialized by the selectbox logic or by loading a chat
    if "qa_chain_instance" not in st.session_state:
        st.session_state.qa_chain_instance = None
    if "default_greeting" not in st.session_state:
        st.session_state.default_greeting = "Welcome! Select a model to begin." # Generic initial
    if "critical_error_message" not in st.session_state:
        st.session_state.critical_error_message = None
    if "current_chat_id" not in st.session_state: # current_chat_id is the ID of the active chat session
        st.session_state.current_chat_id = generate_chat_id()
    if "current_model_initialized" not in st.session_state: # Tracks if RAG components for selected_model are loaded
        st.session_state.current_model_initialized = None
    if "selected_model" not in st.session_state: # Initialize selected_model if not present
        st.session_state.selected_model = st.session_state.model_options[0]


def get_initial_bot_message():
    """Determines the initial bot message based on current global state."""
    model_name = st.session_state.get("selected_model", "the selected model")
    if st.session_state.get("critical_error_message"):
        reason = st.session_state.critical_error_message
        if 'CRITICAL:' in reason:
            reason = reason.split('CRITICAL:')[1].strip()
        return f"Welcome! Chatbot with {model_name} is currently not operational. Reason: {reason}"

    # Use the default_greeting if it's specific and not the generic one
    specific_greeting = st.session_state.get("default_greeting", f"Welcome to the RAG Chatbot (using {model_name})!")
    if specific_greeting and specific_greeting != "Welcome! Select a model to begin.":
         # Ensure the greeting mentions the model if it's a specific RAG-loaded greeting
        if f"(using {model_name})" not in specific_greeting and "RAG Chatbot" in specific_greeting:
             return f"{specific_greeting} (using {model_name})"
        return specific_greeting
    return f"Welcome to the RAG Chatbot (using {model_name})!"


def initialize_chat_session(new_chat_requested=False):
    """Initializes or resets the current chat session messages and ID."""
    # Called when a new chat is explicitly started, or model changes, or on first load.
    if new_chat_requested or "messages" not in st.session_state or not st.session_state.messages or \
       st.session_state.get("current_chat_id") is None:
        st.session_state.current_chat_id = generate_chat_id()
        st.session_state.messages = [{"role": "assistant", "content": get_initial_bot_message()}]
    elif not st.session_state.messages: # Fallback if messages becomes empty unexpectedly
        st.session_state.messages = [{"role": "assistant", "content": get_initial_bot_message()}]


# --- Streamlit App Logic ---

# Renamed and refactored from initialize_rag_components_cached
def initialize_rag_components_for_model(selected_model_type: str):
    """Initializes RAG components for the selected model and stores them in session state."""

    # Reset/initialize state for the current model attempt
    st.session_state.qa_chain_instance = None
    # Basic default greeting, will be refined if RAG init proceeds
    st.session_state.default_greeting = f"Welcome to the RAG Chatbot (using {selected_model_type})!"
    st.session_state.critical_error_message = None
    st.session_state.current_model_initialized = None # Mark as not initialized until success

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
        st.session_state.critical_error_message = f"Error: Unknown model type '{selected_model_type}' selected."
        # default_greeting already set to a generic welcome for this unknown type
        return

    api_key = os.environ.get(api_key_name)
    if not api_key:
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key_val_env = line.strip().split('=', 1)
                        if len(key_val_env) == 2 and key_val_env[0] == api_key_name:
                            api_key = key_val_env[1].strip().strip('"').strip("'")
                            break
        if not api_key:
            st.session_state.critical_error_message = f"CRITICAL: {api_key_name} not found. Chatbot with {model_display_name} cannot operate."
            st.session_state.default_greeting = f"Welcome! API Key for {model_display_name} not configured." # More specific than generic
            return

    # Update default greeting to reflect the model being loaded, even if RAG fails later
    st.session_state.default_greeting = f"Welcome to the RAG Chatbot (using {model_display_name})!"
    try:
        _greeting_from_file, _documents = load_documents()
        # If a GREETING was found in training_data.txt, use it
        if _greeting_from_file and "Welcome to the RAG Chatbot!" not in _greeting_from_file :
            st.session_state.default_greeting = _greeting_from_file

        if not _documents:
            # This is a critical issue if no documents are found for RAG.
            st.session_state.critical_error_message = f"Error: No documents found for RAG with {model_display_name}. Chatbot cannot answer questions based on documents."
            # qa_chain_instance remains None
            return

        _texts = split_documents(_documents)
        if not _texts:
            st.session_state.critical_error_message = f"Error: Documents found but could not be split for {model_display_name}."
            # qa_chain_instance remains None
            return

        st.session_state.qa_chain_instance = setup_rag_pipeline(_texts, api_key, selected_model_type)
        st.session_state.critical_error_message = None # Clear any error if successful
        st.session_state.current_model_initialized = selected_model_type # Mark current model as successfully initialized

    except FileNotFoundError:
        st.session_state.critical_error_message = "Error: The 'training_data.txt' file was not found."
    except Exception as e:
        st.session_state.critical_error_message = f"An unexpected error occurred during RAG setup for {model_display_name}: {str(e)}"
    # qa_chain_instance might still be None if an error occurred. current_model_initialized will not be set.

# --- Sidebar and Chat Management Logic ---
def handle_new_chat():
    """Handles starting a new chat."""
    initialize_chat_session(new_chat_requested=True)
    # Clear any pending loads if a new chat is explicitly started.
    if "pending_chat_load_id" in st.session_state:
        del st.session_state.pending_chat_load_id

def handle_save_chat():
    """Handles saving the current chat to archived_chats."""
    if "messages" not in st.session_state or len(st.session_state.messages) <= 1:
        st.toast("Cannot save an empty or initial chat.", icon="⚠️")
        return

    first_user_message_content = None
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            first_user_message_content = msg["content"]
            break

    if first_user_message_content:
        title = f"Chat: {first_user_message_content[:30]}..."
    else: # Should not happen if len(st.session_state.messages) > 1 and it contains user messages
        title = f"Chat saved at {time.strftime('%Y-%m-%d %H:%M:%S')}"

    archive_id = generate_chat_id() # Unique ID for this archive entry

    # Create a deep copy of messages for archiving
    messages_copy = []
    for msg in st.session_state.messages:
        messages_copy.append(msg.copy())

    st.session_state.archived_chats.append({
        "id": archive_id, # Unique ID for this saved instance
        "title": title,
        "messages": messages_copy,
        "original_chat_id": st.session_state.current_chat_id, # The ID of the chat session when it was saved
        "model_used": st.session_state.selected_model,
        "timestamp": time.time()
    })
    st.toast(f"Chat '{title}' saved!", icon="✅")

def handle_load_chat(archive_id_to_load):
    """Handles loading a chat from archives."""
    chat_to_load = next((c for c in st.session_state.archived_chats if c["id"] == archive_id_to_load), None)

    if not chat_to_load:
        st.error("Failed to load chat. It may have been deleted.")
        return

    # If the model of the chat to load is different from the currently selected model
    if chat_to_load["model_used"] != st.session_state.selected_model:
        st.session_state.selected_model = chat_to_load["model_used"]
        # Store the ID of the chat we intend to load after the model switch and rerun
        st.session_state.pending_chat_load_id = archive_id_to_load
        # Setting selected_model and doing a rerun will trigger RAG re-initialization in main()
        # After re-initialization, logic in main() will check for pending_chat_load_id.
        st.rerun()
    else:
        # Model is the same, just load messages and set current_chat_id
        st.session_state.messages = [msg.copy() for msg in chat_to_load["messages"]]
        st.session_state.current_chat_id = chat_to_load["original_chat_id"]
        st.toast(f"Chat '{chat_to_load['title']}' loaded!", icon="📂")
        if "pending_chat_load_id" in st.session_state: # Clear if it was pending
            del st.session_state.pending_chat_load_id


def handle_delete_chat(archive_id_to_delete):
    """Handles deleting a chat from archives."""
    chat_to_delete = next((c for c in st.session_state.archived_chats if c["id"] == archive_id_to_delete), None)
    if not chat_to_delete: return

    st.session_state.archived_chats = [c for c in st.session_state.archived_chats if c["id"] != archive_id_to_delete]
    st.toast(f"Chat '{chat_to_delete['title']}' deleted.", icon="🗑️")

    # Optional: If the deleted chat was the one currently displayed, start a new chat.
    # This requires comparing message content or original_chat_id if it matches current_chat_id.
    # For simplicity, if the current_chat_id matches the original_chat_id of the deleted chat, then start new.
    if st.session_state.current_chat_id == chat_to_delete["original_chat_id"]:
         # Check if the messages are also identical to be more certain it's the *exact* loaded chat
        if st.session_state.messages == chat_to_delete["messages"]:
            handle_new_chat() # This will also trigger a rerun via its own logic or by adding st.rerun() there
            st.rerun() # Ensure UI updates to the new chat state

def render_sidebar_chat_management():
    """Renders chat management UI in the sidebar."""
    st.sidebar.title("Chat Management")
    if st.sidebar.button("➕ New Chat", key="new_chat_button", help="Start a new conversation"):
        handle_new_chat()
        st.rerun()

    # Disable save if only initial bot message exists or no user messages yet
    can_save = ("messages" in st.session_state and
                len(st.session_state.messages) > 1 and
                any(msg["role"] == "user" for msg in st.session_state.messages))

    if st.sidebar.button("💾 Save Current Chat", key="save_chat_button", disabled=not can_save, help="Save the current conversation"):
        handle_save_chat()

    if st.session_state.archived_chats:
        st.sidebar.subheader("Archived Chats")
        # Sort chats by timestamp descending (newest first)
        sorted_chats = sorted(st.session_state.archived_chats, key=lambda c: c.get("timestamp", 0), reverse=True)

        for i, archived_chat in enumerate(sorted_chats):
            # Use columns for layout: Title, Load, Delete
            col_title, col_load, col_delete = st.sidebar.columns([0.7, 0.15, 0.15])

            with col_title:
                col_title.caption(f"{archived_chat['title']} ({archived_chat['model_used']})")
            with col_load:
                if col_load.button("📂", key=f"load_{archived_chat['id']}_{i}", help="Load chat"):
                    handle_load_chat(archived_chat['id'])
                    # handle_load_chat might do its own rerun if model changes.
                    # If not, we might need one here. For now, handle_load_chat manages rerun.
                    if chat_to_load["model_used"] == st.session_state.selected_model : # if model didn't change, force rerun
                        st.rerun()

            with col_delete:
                if col_delete.button("🗑️", key=f"delete_{archived_chat['id']}_{i}", help="Delete chat"):
                    handle_delete_chat(archived_chat['id'])
                    # handle_delete_chat might do its own rerun if current chat is deleted.
                    # If not, we need one here.
                    st.rerun() # Rerun to reflect deletion in the list
    else:
        st.sidebar.caption("No saved chats yet.")

# --- Main Application ---
# --- Main Application ---
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("📄 RAG Powered Chatbot")

    initialize_global_state() # Sets up all session_state keys if not present

    # --- Model Selection Logic ---
    st.sidebar.title("⚙️ Configuration")

    # Determine current index for selectbox based on session state
    try:
        current_model_index = st.session_state.model_options.index(st.session_state.selected_model)
    except (ValueError, AttributeError): # If selected_model not in options or not set
        current_model_index = 0
        st.session_state.selected_model = st.session_state.model_options[0]

    selected_model_from_sidebar = st.sidebar.selectbox(
        "Choose LLM Provider:",
        options=st.session_state.model_options,
        index=current_model_index,
        key="model_selector_widget" # Unique key for the widget itself
    )
    st.sidebar.info(f"Current Model: **{st.session_state.selected_model}**")

    # --- RAG Initialization and Chat Session Management based on Model Selection ---
    # If user changes model via selectbox OR if current model's RAG components aren't initialized yet
    if st.session_state.selected_model != selected_model_from_sidebar or \
       st.session_state.current_model_initialized != st.session_state.selected_model:

        st.session_state.selected_model = selected_model_from_sidebar # Update state

        # If a chat load was pending for the *old* model, it's no longer relevant.
        if "pending_chat_load_id" in st.session_state:
            # Check if pending load was for the NEWLY selected model. If so, keep it.
            pending_chat_info = next((c for c in st.session_state.archived_chats if c["id"] == st.session_state.pending_chat_load_id), None)
            if not pending_chat_info or pending_chat_info["model_used"] != st.session_state.selected_model:
                del st.session_state.pending_chat_load_id

        initialize_rag_components_for_model(st.session_state.selected_model) # Initialize for current st.session_state.selected_model

        # If RAG init was successful AND no chat load is pending for this new model, start a fresh chat.
        # If RAG init failed, get_initial_bot_message will reflect that.
        if "pending_chat_load_id" not in st.session_state:
             initialize_chat_session(new_chat_requested=True)
        st.rerun() # Rerun to apply changes, handle pending load if any, and update UI.

    # --- Handle Pending Chat Load (e.g., after a model switch that required a rerun) ---
    if "pending_chat_load_id" in st.session_state:
        archive_id_to_load = st.session_state.pending_chat_load_id
        chat_to_load_details = next((c for c in st.session_state.archived_chats if c["id"] == archive_id_to_load), None)

        if chat_to_load_details and \
           chat_to_load_details["model_used"] == st.session_state.selected_model:
            # Target model for the chat is the currently selected model
            if st.session_state.current_model_initialized == st.session_state.selected_model:
                # RAG components are ready for this model
                st.session_state.messages = [msg.copy() for msg in chat_to_load_details["messages"]]
                st.session_state.current_chat_id = chat_to_load_details["original_chat_id"]
                st.toast(f"Chat '{chat_to_load_details['title']}' loaded!", icon="📂")
                del st.session_state.pending_chat_load_id
            else:
                # RAG components failed to initialize for this model
                st.warning(f"Could not load chat for '{st.session_state.selected_model}' as the model could not be initialized. Displaying a new chat instead.")
                del st.session_state.pending_chat_load_id
                initialize_chat_session(new_chat_requested=True)
                st.rerun() # Rerun to show new chat for the failed model
        elif not chat_to_load_details:
            # The chat to load was not found (e.g., deleted)
            del st.session_state.pending_chat_load_id
            initialize_chat_session(new_chat_requested=True)
            st.rerun()
        # If chat_to_load_details["model_used"] != st.session_state.selected_model, a rerun was already triggered
        # by the model selection logic, so this block will be re-evaluated after that rerun.

    # --- Ensure chat session is initialized if it somehow isn't (e.g., very first run) ---
    if "messages" not in st.session_state or not st.session_state.messages:
        initialize_chat_session(new_chat_requested=True)

    # --- Render Sidebar for Chat Management ---
    render_sidebar_chat_management()

    # --- Display Current Chat Context ---
    active_chat_display_title = "New Conversation" # Default title
    # Check if the current chat (by original_chat_id) corresponds to any loaded archived chat's title
    loaded_chat_title = None
    for chat in st.session_state.archived_chats:
        if chat.get("original_chat_id") == st.session_state.current_chat_id:
            # Further check if messages are identical to confirm it's not a modified version of a loaded chat
            # This is a simple check. For true "is loaded and unmodified" status, a more robust flag might be needed.
            if st.session_state.messages == chat.get("messages"):
                loaded_chat_title = chat.get("title")
                break

    if loaded_chat_title:
        active_chat_display_title = f"Loaded: \"{loaded_chat_title}\""

    short_chat_id_display = st.session_state.current_chat_id.split('-')[0]
    st.markdown(f"**Current Chat:** {active_chat_display_title} `(ID: ...{short_chat_id_display})`")

    # --- Display Global Error Messages (related to RAG setup for the current model) ---
    if st.session_state.critical_error_message and not st.session_state.qa_chain_instance:
        # This error is prominent if the RAG system itself is non-operational
        # The initial message in the chat window (from get_initial_bot_message) also reflects this.
        st.error(f"RAG System Error: {st.session_state.critical_error_message}")

    # --- Display Chat Messages ---
    # Ensure messages list exists and is a list (it should always be by now)
    if isinstance(st.session_state.get("messages"), list):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        # Fallback if messages is somehow not a list (should not happen)
        st.session_state.messages = [{"role": "assistant", "content": get_initial_bot_message()}]
        with st.chat_message("assistant"):
            st.markdown(st.session_state.messages[0]["content"])


    # --- Handle User Input ---
    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            full_response_text = ""

            if not st.session_state.qa_chain_instance:
                error_to_show = st.session_state.critical_error_message if st.session_state.critical_error_message else "Chatbot is not properly initialized for the selected model."
                full_response_text = f"Cannot process query: {error_to_show}"
                message_placeholder.markdown(f":red[Error: {full_response_text}]")
            else:
                try:
                    response = st.session_state.qa_chain_instance.invoke({"query": prompt})
                    answer = response.get("result", "Sorry, I could not find an answer based on the documents.")
                    full_response_text = answer

                    if response.get("source_documents"):
                        full_response_text += "\n\n**Sources:**"
                        for i, doc in enumerate(response["source_documents"]):
                            source_info = f"Source {i+1}: "
                            if 'source' in doc.metadata:
                                source_info += f"File: {os.path.basename(doc.metadata['source'])}"
                            if 'start_index' in doc.metadata:
                                 source_info += f", Start Index: {doc.metadata['start_index']}"
                            full_response_text += f"\n\n*{source_info}*\n>{doc.page_content[:200]}..."
                    message_placeholder.markdown(full_response_text)
                except Exception as e:
                    full_response_text = f"Error processing your query: {str(e)}"
                    message_placeholder.markdown(f":red[Error: {full_response_text}]")

            st.session_state.messages.append({"role": "assistant", "content": full_response_text})

if __name__ == '__main__':
    # Minimal .env loading for local development.
    # Critical API keys should ideally be set as environment variables directly.
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key_val = line.strip().split('=', 1)
                        if len(key_val) == 2:
                            # Set them as environment variables so os.environ.get() can pick them up
                            # This is done before any Streamlit/Langchain code that might need them runs.
                            os.environ[key_val[0]] = key_val[1].strip().strip('"').strip("'")
        except Exception:
            # Silently ignore .env loading errors, relying on direct env vars or later error handling.
            pass
    main()

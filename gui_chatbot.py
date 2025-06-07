# gui_chatbot.py
import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox
import os
import threading

# --- OpenAI API Key Setup ---
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    if 'OPENAI_API_KEY' in os.environ:
        OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    # No explicit else here to show error immediately, initialize_rag_components will handle it.

# --- RAG Components ---
# Adapted from rag_chatbot.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma, FAISS # Added FAISS

qa_chain = None
# greeting_message is loaded by load_documents, default provided in function
greeting_message = "Welcome to the RAG Chatbot GUI! Initializing..." # Default before loading

# --- Functions adapted from rag_chatbot.py ---
def load_documents(filepath="training_data.txt"):
    loader = TextLoader(filepath, encoding='utf-8') # Specify encoding
    all_docs = loader.load()

    loaded_greeting_message = "Welcome to the RAG Chatbot!" # Default greeting
    processed_documents = []

    if all_docs:
        first_doc_content = all_docs[0].page_content
        if first_doc_content.startswith("GREETING: "):
            loaded_greeting_message = first_doc_content[len("GREETING: "):].strip()
            if len(all_docs) > 1:
                processed_documents = all_docs[1:]
        else:
            processed_documents = all_docs

    # Update global greeting_message for the GUI to use
    global greeting_message
    greeting_message = loaded_greeting_message
    return processed_documents # Return only documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    texts = text_splitter.split_documents(documents)
    return texts

def setup_rag_pipeline(texts, openai_api_key_param): # Renamed parameter to avoid conflict
    print("Initializing embeddings model...") # Keep print for console feedback during init
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key_param)

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
            messagebox.showerror("Vector Store Error", "Could not initialize any suitable vector store. Check dependencies (e.g., tiktoken for Chroma, or FAISS install) and environment.")
            raise RuntimeError("Failed to initialize vector store") from e_faiss # Re-raise for initialize_rag_components

    if not vectorstore:
        messagebox.showerror("Vector Store Error", "Vector store initialization failed unexpectedly.")
        raise RuntimeError("Vector store initialization failed.")

    print("Initializing LLM...")
    llm = OpenAI(openai_api_key=openai_api_key_param)

    print("Creating retriever...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("Creating RAG chain...")
    new_qa_chain = RetrievalQA.from_chain_type( # Assign to new_qa_chain then global
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False # GUI might not want verbose chain output to console
    )
    print("RAG pipeline setup complete.")
    return new_qa_chain


def initialize_rag_components():
    """
    Initializes RAG components: API key, documents, RAG chain.
    Returns True on success, False on failure.
    """
    global qa_chain, OPENAI_API_KEY, greeting_message

    if not OPENAI_API_KEY:
        # Attempt to get API key again, e.g., if set after script start but before init
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        if not OPENAI_API_KEY and 'OPENAI_API_KEY' in os.environ:
            OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

        if not OPENAI_API_KEY:
            messagebox.showerror("API Key Error", "OPENAI_API_KEY environment variable not found. Please set it and restart.")
            return False

    try:
        print("GUI: Loading documents...")
        # load_documents now updates global greeting_message directly
        documents = load_documents() # Default filepath="training_data.txt"

        if not documents:
            print("GUI: No documents found or loaded from training_data.txt (excluding GREETING).")
            # Allow proceeding if only a greeting is present, chat will just not have context.
            # Or show a warning:
            messagebox.showwarning("No Documents", "No documents found in 'training_data.txt'. The chatbot will operate without external knowledge.")
            # If we want to prevent startup without documents:
            # messagebox.showerror("Data Error", "No documents loaded. Please check 'training_data.txt'.")
            # return False
            texts = [] # Ensure texts is an empty list if no documents
        else:
            print("GUI: Splitting documents...")
            texts = split_documents(documents)
            if not texts:
                messagebox.showwarning("Document Processing Error", "Document splitting resulted in no text chunks. Chatbot may not have context.")
                # Allow proceeding, RAG chain might handle empty texts gracefully or error later.
                # If we want to prevent startup:
                # messagebox.showerror("Data Error", "Failed to split documents into usable chunks.")
                # return False

        # Only setup RAG pipeline if there are texts. Or let setup_rag_pipeline handle empty texts if it can.
        # For now, let's assume setup_rag_pipeline needs texts.
        if texts:
            print("GUI: Setting up RAG pipeline...")
            qa_chain = setup_rag_pipeline(texts, OPENAI_API_KEY)
        else:
            # If no texts, qa_chain remains None.
            # We could have a dummy chain or specific handling if desired.
            # For now, if no texts, no RAG chain. User will be informed by "RAG chain not initialized".
            print("GUI: No texts to process, RAG chain will not be initialized with document context.")
            # greeting_message would have been set by load_documents regardless

        print(f"GUI: Initialization complete. Greeting: {greeting_message}")
        return True

    except FileNotFoundError:
        messagebox.showerror("File Error", f"The data file (e.g., training_data.txt) was not found. Please create it.")
        return False
    except RuntimeError as e: # Catch specific RuntimeError from setup_rag_pipeline
        print(f"GUI: RAG Initialization Runtime Error: {e}")
        # Error already shown by setup_rag_pipeline or vector store part
        return False
    except Exception as e:
        print(f"GUI: An unexpected error occurred during RAG initialization: {e}")
        messagebox.showerror("Initialization Error", f"An unexpected error occurred: {e}")
        return False


class ChatApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG Chatbot")
        self.geometry("700x500")

        self.chat_area = None
        self.input_field = None
        self.send_button = None

        # Initialize RAG components in a separate thread to avoid freezing GUI
        # and then create UI components.
        self.initialize_and_create_ui()

    def initialize_and_create_ui(self):
        # Show a loading message or disable input until initialization is done
        # For simplicity, we'll proceed and let initialize_rag_components show errors if any.

        # Run initialization in a new thread to keep UI responsive
        init_thread = threading.Thread(target=self.finish_initialization)
        init_thread.daemon = True # Ensure thread exits when main program exits
        init_thread.start()

    def finish_initialization(self):
        if initialize_rag_components():
            self.after(0, self.create_widgets)
            # greeting_message is now a global updated by load_documents
            self.after(0, lambda: self.display_message(f"Bot: {greeting_message}\n"))
        else:
            # Error messages are now more specific within initialize_rag_components
            # We can add a generic failure message here or rely on the specific ones.
            self.after(0, self.handle_initialization_failure)

    def handle_initialization_failure(self):
        # Check if chat_area exists before trying to update it
        if self.chat_area:
            self.display_message("Bot: Initialization failed. Please check console/logs and API key/data.\n")
        else: # If UI hasn't been created, show a popup.
            messagebox.showerror("Fatal Error", "Initialization failed. Chatbot cannot start. Check API key and data file.")
        # Optionally, disable input fields if they exist or close app
        if self.input_field: self.input_field.config(state=tk.DISABLED)
        if self.send_button: self.send_button.config(state=tk.DISABLED)
        # self.destroy() # Uncomment to close app on failure

    def create_widgets(self):
        # Chat display area
        self.chat_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 10))
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Input frame
        input_frame = tk.Frame(self)
        input_frame.pack(padx=10, pady=(0,10), fill=tk.X, expand=False)

        # Input field
        self.input_field = tk.Entry(input_frame, font=("Arial", 10))
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.input_field.bind("<Return>", self.send_message_event) # Bind Enter key

        # Send button
        self.send_button = tk.Button(input_frame, text="Send", command=self.send_message_event, font=("Arial", 10))
        self.send_button.pack(side=tk.RIGHT, padx=(5,0))

    def send_message_event(self, event=None): # Added event=None for a/s bind
        query = self.input_field.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a question.")
            return

        self.display_message(f"You: {query}\n")
        self.input_field.delete(0, tk.END)

        # Disable input while processing
        self.input_field.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, "Bot: Thinking...\n")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)


        # Process query in a separate thread to avoid freezing GUI
        query_thread = threading.Thread(target=self.process_query_thread, args=(query,))
        query_thread.daemon = True
        query_thread.start()

    def process_query_thread(self, query):
        global qa_chain
        response_message = ""
        try:
            if qa_chain:
                result = qa_chain.invoke({"query": query})
                answer = result.get("result", "No answer found.")
                response_message = f"Bot: {answer}\n"

                if result.get("source_documents"):
                    response_message += "Sources:\n"
                    for i, doc in enumerate(result["source_documents"]):
                        source_info = f"  Source {i+1}: "
                        if 'source' in doc.metadata:
                            source_info += f"File: {os.path.basename(doc.metadata['source'])}" # Show only filename
                        if 'start_index' in doc.metadata:
                             source_info += f", Start Index: {doc.metadata['start_index']}"
                        response_message += f"{source_info}\n" # Removed snippet for brevity in GUI
            elif not OPENAI_API_KEY: # Check if API key was the issue for qa_chain being None
                response_message = "Bot: Error - RAG chain not available (API key might be missing or invalid).\n"
            else: # qa_chain is None for other reasons (e.g. no documents, setup failed)
                response_message = "Bot: Error - RAG chain not initialized. This could be due to missing/empty data file or other setup issues.\n"
        except Exception as e:
            response_message = f"Bot: Error processing your query: {e}\n"

        # Schedule UI update back on the main thread
        self.after(0, self.update_chat_after_processing, response_message)

    def update_chat_after_processing(self, response_message):
        # Remove "Thinking..." message more robustly
        self.chat_area.config(state=tk.NORMAL)

        # Get all content
        current_content = self.chat_area.get("1.0", tk.END)

        # Find the last occurrence of "Bot: Thinking..."
        # We need to search backwards from the end of the text.
        # The text widget inserts a newline at the very end, so ignore that.
        if current_content.endswith("\n"):
            current_content_for_search = current_content[:-1]
        else:
            current_content_for_search = current_content

        thinking_marker = "Bot: Thinking...\n"
        last_thinking_pos = current_content_for_search.rfind(thinking_marker)

        if last_thinking_pos != -1:
            # Convert flat string index to text widget index (line.char)
            # Count lines up to last_thinking_pos
            num_lines = current_content_for_search[:last_thinking_pos].count('\n') + 1
            char_pos_on_line = last_thinking_pos - current_content_for_search.rfind("\n", 0, last_thinking_pos) -1

            # Define the range to delete: from the start of "Bot: Thinking..." to the end of that line.
            # The 'end' of a line in Text widget includes the newline.
            start_delete_index = f"{num_lines}.{char_pos_on_line}"
            # We want to delete the "Bot: Thinking..." line.
            # So we find the end of that line.
            end_marker_search_start = last_thinking_pos + len(thinking_marker)
            next_newline_after_marker = current_content.find("\n", end_marker_search_start)

            if next_newline_after_marker != -1 :
                # if "Bot: Thinking..." was not the very last thing
                # We delete from start_delete_index up to the start of the next line.
                # Example: delete from "23.0" to "24.0" to delete line 23.
                # Or, delete from "23.0" to "23.end" to delete the content of line 23.
                # Let's try deleting the line content where "Bot: Thinking..." starts, up to its newline.
                end_delete_index = f"{num_lines}.{char_pos_on_line + len(thinking_marker)}"
                self.chat_area.delete(start_delete_index, end_delete_index)
            else:
                # If "Bot: Thinking..." is the last content (potentially without its own newline yet if typing fast)
                # This case is less likely given how we add it with \n
                self.chat_area.delete(start_delete_index, tk.END)

        self.display_message(response_message)

        # Re-enable input
        self.input_field.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.input_field.focus()


    def display_message(self, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, message)
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END) # Scroll to the end


if __name__ == '__main__':
    app = ChatApplication()
    app.mainloop()

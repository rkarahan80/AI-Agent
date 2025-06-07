# rag_tool.py
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma, FAISS

_cached_rag_chain = None
_cached_api_key = None
_cached_training_data_path = None

def load_documents_for_tool(filepath="training_data.txt"):
    loader = TextLoader(filepath, encoding='utf-8')
    documents = loader.load()
    return documents

def split_documents_for_tool(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    texts = text_splitter.split_documents(documents)
    return texts

def setup_rag_pipeline_for_tool(texts, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = None
    try:
        vectorstore = Chroma.from_documents(texts, embeddings)
    except Exception:
        try:
            vectorstore = FAISS.from_documents(texts, embeddings)
        except Exception as e_faiss:
            raise RuntimeError(f"Failed to initialize vector store: {e_faiss}")

    if not vectorstore:
        raise RuntimeError("Vector store initialization failed.")

    llm = OpenAI(openai_api_key=openai_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )
    return qa_chain

def query_rag_chain(user_query: str, training_data_path: str = "training_data.txt") -> str:
    """Answers query based on documents from training_data_path using RAG.
    Args:
        user_query (str): User's question.
        training_data_path (str): Path to knowledge base file. Defaults to "training_data.txt".
    Returns:
        str: Answer from RAG pipeline or error message.
    """
    global _cached_rag_chain, _cached_api_key, _cached_training_data_path

    current_openai_api_key = os.getenv('OPENAI_API_KEY')
    if not current_openai_api_key:
        if os.path.exists("rag_adk_agent/.env"):
            with open("rag_adk_agent/.env", "r") as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        current_openai_api_key = line.strip().split('=')[1].strip('\"').strip("'")
                        break
    if not current_openai_api_key:
        return "Error: OPENAI_API_KEY not found."

    try:
        if (_cached_rag_chain is None or
            _cached_api_key != current_openai_api_key or
            _cached_training_data_path != training_data_path):

            documents = load_documents_for_tool(filepath=training_data_path)
            if not documents:
                return f"Error: No documents from '{training_data_path}'."

            texts = split_documents_for_tool(documents)
            if not texts:
                return "Error: No text chunks after splitting."

            _cached_rag_chain = setup_rag_pipeline_for_tool(texts, current_openai_api_key)
            _cached_api_key = current_openai_api_key
            _cached_training_data_path = training_data_path

        if not _cached_rag_chain:
             return "Error: RAG chain not initialized."

        result = _cached_rag_chain.invoke({"query": user_query})
        answer = result.get("result", "Error: No result in RAG output.")
        return answer

    except FileNotFoundError:
        return f"Error: Training data '{training_data_path}' not found."
    except RuntimeError as e:
        return f"Error in RAG processing: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

import os
import logging
from typing import List
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage

# Configuration Constants
PDF_PATH = "harrypotter.pdf"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMResponse:
    """ 
        Minimal abstractions for an LLM response.
        This keeps the example self-contained while still allowing us to attach 
        token usage, and other metadata for observability
    """
    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize lightweight local HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def setup_vector_store() -> int:
    """
    Extracts text from harrypotter.pdf, splits into chunks, converts to 
    embeddings using a lightweight model, and stores in a local Chroma DB.
    """
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Missing {PDF_PATH} in current directory.")

    logger.info(f"Loading PDF from {PDF_PATH} using PyPDFLoader...")
    # 1. Load PDF lazily to prevent parsing hundreds of pages
    loader = PyPDFLoader(PDF_PATH)
    
    # Just take the first 3 pages that actually contain text
    docs = []
    for doc in loader.lazy_load():
        if len(doc.page_content.strip()) > 10:
            docs.append(doc)
            logger.info(f"Extracted page {doc.metadata.get('page', 'unknown')} with {len(doc.page_content)} characters.")
        if len(docs) >= 3:
            break

    if not docs:
        logger.error("No extractable text found in the PDF!")
        raise ValueError("No extractable text found in the PDF.")

    logger.info(f"Splitting {len(docs)} pages into chunks...")
    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    if not splits:
        logger.error("Splitting resulted in 0 chunks!")
        raise ValueError("Could not create text chunks.")

    logger.info(f"Created {len(splits)} chunks. Converting to embeddings and saving to ChromaDB...")
    # 3. Create / Update Vectorstore with embeddings
    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=CHROMA_DIR
    )
    
    logger.info("Vector store setup is complete!")
    return len(splits)


async def retrieve_documents(query: str, top_k: int = 5) -> List[str]:
    """ 
        RAG Document Retrieval
        Queries the Chroma database for relevant context chunks.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    # Run synchronously as standard retriever.invoke
    docs = retriever.invoke(query)
    
    return [doc.page_content for doc in docs]


async def generate_llm_response(prompt: str) -> LLMResponse:
    """ 
        Actual LLM API call using Langchain Groq SDK.
    """
    # Initialize the Groq Chat Model. 
    # Ensure GROQ_API_KEY is available in your environment variables.
    llm = ChatGroq(model_name=GROQ_MODEL, temperature=0.7)
    
    ai_msg: AIMessage = await llm.ainvoke(prompt)
    
    # Extract token usage
    usage = ai_msg.response_metadata.get("token_usage", {})
    prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
    completion_tokens = usage.get("completion_tokens", len(str(ai_msg.content).split()))
    
    return LLMResponse(str(ai_msg.content), prompt_tokens, completion_tokens)

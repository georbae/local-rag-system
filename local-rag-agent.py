import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
import docx2txt
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import logging
from tqdm import tqdm
import concurrent.futures

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Streamlit app
st.title("Local RAG System")

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def process_docx(file_path):
    try:
        text = docx2txt.process(file_path)
        return text
    except Exception as e:
        logger.error(f"Error processing DOCX {file_path}: {str(e)}")
        return ""

def process_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text = ""
        for page in pages:
            text += page.page_content + "\n"
        
        # OCR for images in PDF
        images = convert_from_path(file_path)
        for image in images:
            text += extract_text_from_image(image) + "\n"
        
        return text
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        return ""

def process_document(file_path):
    if file_path.endswith('.pdf'):
        return process_pdf(file_path)
    elif file_path.endswith('.docx'):
        return process_docx(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_path}")
        return ""

@st.cache_resource
def initialize_rag_system():
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Load documents
        directory = os.getenv('DIRECTORY')
        logger.info(f"Attempting to load documents from: {directory}")
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory) 
                      if f.endswith(('.pdf', '.docx'))]
        
        if not file_paths:
            logger.warning(f"No PDF or DOCX files found in {directory}")
            return None

        # Process documents in parallel
        documents = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(process_document, file_path): file_path 
                              for file_path in file_paths}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                file_path = future_to_file[future]
                try:
                    text = future.result()
                    if text:
                        documents.append({"content": text, "source": file_path})
                        status_text.text(f"Processed {os.path.basename(file_path)}")
                    else:
                        logger.warning(f"No text extracted from {file_path}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                progress_bar.progress((i + 1) / len(file_paths))

        if not documents:
            logger.warning(f"No documents could be loaded from {directory}")
            return None

        # Split documents
        status_text.text("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.create_documents([doc["content"] for doc in documents],
                                                    metadatas=[{"source": doc["source"]} for doc in documents])

        # Create embeddings
        status_text.text("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize or load Chroma DB
        chroma_db_path = os.getenv('CHROMA_DB_PATH')
        if not os.path.exists(chroma_db_path):
            os.makedirs(chroma_db_path, exist_ok=True)

        status_text.text("Initializing vector store...")
        vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=chroma_db_path)
        vectorstore.persist()
        st.info(f"Vector store initialized at {chroma_db_path}")

                # Initialize local LLM
        status_text.text("Initializing LLM...")
        model_name = os.getenv('LLM_MODEL')
        model_path = os.path.join(os.getenv('MODEL_PATH', './models'), model_name)

        # Check if the model file exists
        if not os.path.exists(model_path):
            # Try alternative paths
            alternative_paths = [
                os.path.join('./models', model_name),
                os.path.join('.', model_name),
                model_name  # In case it's an absolute path
            ]
            for path in alternative_paths:
                if os.path.exists(path):
                    model_path = path
                    logger.info(f"Model found in alternative path: {model_path}")
                    break
            else:
                raise FileNotFoundError(f"Model file not found. Tried paths: {model_path}, {', '.join(alternative_paths)}")

        logger.info(f"Loading model from: {model_path}")
        print(f"Attempting to load model from: {model_path}")

        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.75,
            max_tokens=2000,
            top_p=1,
            n_ctx=2048,
            n_batch=8,
            verbose=True,
        )


        # Create a retrieval chain
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

        status_text.text("RAG system initialized successfully!")
        progress_bar.progress(100)

        return qa_chain

    except Exception as e:
        logger.exception("Error initializing RAG system")
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

# Initialize or load the RAG system
qa_chain = initialize_rag_system()

if qa_chain:
    # User input
    query = st.text_input("Enter your question:")

    if st.button("Submit"):
        if query:
            with st.spinner("Generating answer..."):
                result = qa_chain({"query": query})
                st.write(result['result'])
        else:
            st.warning("Please enter a question.")

    # Display information about the system
    st.sidebar.title("System Information")
    st.sidebar.info(f"Document Directory: {os.getenv('DIRECTORY')}")
    st.sidebar.info(f"Chroma DB Path: {os.getenv('CHROMA_DB_PATH')}")
    st.sidebar.info(f"Model: {os.getenv('LLM_MODEL')}")
else:
    st.error("Failed to initialize the RAG system. Please check the error messages above.")

if __name__ == "__main__":
    # You can add any additional setup or tests here
    pass

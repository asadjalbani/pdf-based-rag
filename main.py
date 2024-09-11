import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
import json
import logging
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
os.getenv('GROQ_API_KEY')

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_and_split_pdf(pdf_path):
    try:
        # Check if the PDF file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The PDF file does not exist at path: {pdf_path}")

        # Load the PDF
        logging.info(f"Loading PDF from {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        logging.info(f"Successfully loaded {len(pages)} pages from the PDF")

        # Split the text
        logging.info("Splitting the document into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(pages)
        
        if not docs:
            raise logging.info("Document splitting resulted in an empty list of documents.")
        logging.info(f"Successfully split the document into {len(docs)} chunks")

        # Prepare the output directory
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        output_text_file_path = os.path.join(output_dir, 'text.txt')

        # Save the extracted text chunks to a text file
        logging.info(f"Saving extracted text to {output_text_file_path}")
        with open(output_text_file_path, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(docs):
                try:
                    f.write(f"--- Document {i + 1} ---\n")
                    f.write(doc.page_content)
                    f.write("\n\n")
                except Exception as e:
                    logging.error(f"Error writing chunk {i + 1} to file: {e}")

        logging.info(f"Text extracted and saved to {output_text_file_path}")
        return docs

    except FileNotFoundError as e:
        logging.error(f"File not found error: {e}")
    except ValueError as e:
        logging.error(f"Value error (possibly due to corrupt PDF): {e}")
    except PermissionError as e:
        logging.error(f"Permission error (check file access rights): {e}")
    except Exception as e:
        logging.error(f"Unexpected error in load_and_split_pdf: {e}")
    
    return None  # Return None if an error occured

def create_or_load_vector_store(docs, vector_store_path, pdf_path):
    if not docs:
        logging.error("No documents provided to create or load vector store.")
        return None  # Early return if docs is None or empty
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create a unique identifier for the file based on its path and modification time
        file_stats = os.stat(pdf_path)
        file_identifier = f"{pdf_path}_{file_stats.st_mtime}"
        
        # Path for the metadata file
        metadata_path = os.path.join(vector_store_path, "metadata.json")
        
        current_time = datetime.now()
        
        # Load or create metadata
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
        except json.JSONDecodeError:
            logging.error("Error reading metadata file. Creating new metadata.")
            metadata = {}
        
        # Check if we have valid embeddings for this file
        if file_identifier in metadata:
            embedding_info = metadata[file_identifier]
            embedding_time = datetime.fromisoformat(embedding_info['timestamp'])
            if current_time - embedding_time < timedelta(hours=24):
                logging.info("Loading existing vector store...")
                try:
                    return FAISS.load_local(embedding_info['path'], embeddings, allow_dangerous_deserialization=True)
                except Exception as e:
                    logging.error(f"Error loading existing vector store: {e}")
                    logging.info("Creating new vector store due to loading error.")
        
        # If we're here, we need to create new embeddings
        logging.info("Creating new vector store...")
        try:
            vector_store = FAISS.from_documents(docs, embeddings)
        except Exception as e:
            logging.error(f"Error creating new vector store: {e}")
            raise
        
        # Save the new vector store
        new_store_path = os.path.join(vector_store_path, f"store_{int(time.time())}")
        try:
            vector_store.save_local(new_store_path)
        except Exception as e:
            logging.error(f"Error saving new vector store: {e}")
            raise
        
        # Update metadata
        metadata[file_identifier] = {
            'path': new_store_path,
            'timestamp': current_time.isoformat()
        }
        
        # Save updated metadata
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            logging.error(f"Error saving metadata: {e}")
        
        # Clean up old embeddings
        for fid, info in list(metadata.items()):
            try:
                embedding_time = datetime.fromisoformat(info['timestamp'])
                if current_time - embedding_time > timedelta(hours=24):
                    logging.info(f"Removing old embeddings for {fid}")
                    os.remove(info['path'])
                    del metadata[fid]
            except Exception as e:
                logging.error(f"Error during cleanup of old embeddings: {e}")
        
        # Save cleaned metadata
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            logging.error(f"Error saving cleaned metadata: {e}")
        
        return vector_store
    
    except Exception as e:
        logging.error(f"Unexpected error in create_or_load_vector_store: {e}")
        raise

def setup_rag_chain(vector_store):
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant tasked with answering questions based on the provided document. Your goal is to give direct and concise answers to the user's questions without adding explanations or references about where the information was found. If the answer is not in the document, simply state 'I don't know.' Do not provide any additional context or details beyond the direct answer."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return rag_chain

def ask_question(rag_chain, question):
    start_time = time.time()
    result = rag_chain.invoke({"query": question})
    end_time = time.time()
    
    print(result['result'])
    print(f"Time taken to execute rag_chain: {end_time - start_time} seconds")
    

def main():
    # pdf_path = r"C:\Users\ESHOP\Downloads\Eric Matthes - Python Crash Course_ A Hands-On, Project-Based Introduction to Programming-No Starch .pdf"
    pdf_path = "chapter3.pdf"
    vector_store_path = "document_embeddings"

    docs = load_and_split_pdf(pdf_path) #if not os.path.exists(vector_store_path) else None
    if not docs:
        logging.error("Failed to load documents from the PDF.")
        return  # Exit if no documents loaded

    vector_store = create_or_load_vector_store(docs, vector_store_path, pdf_path)
    if not vector_store:
        logging.error("Failed to create or load vector store.")
        return  # Exit if vector store creation failed

    rag_chain = setup_rag_chain(vector_store)

    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        ask_question(rag_chain, question)
if __name__ == "__main__":
    main()
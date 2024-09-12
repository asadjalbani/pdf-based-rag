import os
import time
import logging
from datetime import datetime, timedelta
from django.core.files.storage import default_storage
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from .models import PDFDocument

# Assuming 'document_path.txt' stores the path to the uploaded PDF
# PDF_PATH_FILE = f'document_path_{int(time.time())}.txt'
User = get_user_model()

def load_and_split_pdf(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The PDF file does not exist at path: {pdf_path}")

        logging.info(f"Loading PDF from {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        logging.info(f"Successfully loaded {len(pages)} pages from the PDF")

        logging.info("Splitting the document into chunks")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)
        
        if not docs:
            raise ValueError("Document splitting resulted in an empty list of documents.")
        logging.info(f"Successfully split the document into {len(docs)} chunks")

        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        output_text_file_path = os.path.join(output_dir, 'text.txt')

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
    
    return None


def create_or_load_user_vector_store(user):
    vector_store_path = f'document_embeddings/{user.id}'

    try:
        # Check if vector store already exists
        if os.path.exists(vector_store_path):
            logging.info("Loading existing user vector store...")
            return FAISS.load_local(vector_store_path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
        else:
            # Return None if no vector store exists yet, will be created later when processing documents
            logging.info("No vector store found, will create new one.")
            return None

    except Exception as e:
        logging.error(f"Error creating or loading vector store: {e}")
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

class RAGView(APIView):
    permission_classes = [IsAuthenticated]

class RAGView(APIView):
    permission_classes = [IsAuthenticated]
    
    # POST: Handle document uploads and update the vector store
    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        user = request.user
        file_path = default_storage.save(f'tmp/{file.name}', ContentFile(file.read()))

        try:

            # Load or create the user-specific vector store
            vector_store = create_or_load_user_vector_store(user)
            
            # Store PDF metadata in the database
            pdf_doc, created = PDFDocument.objects.get_or_create(
                file_path=file_path,
                user=user
            )

            # Process the document and split it into chunks
            docs = load_and_split_pdf(file_path)
            if not docs:
                return Response({"error": "Failed to process the document"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # If vector store exists, update it. Otherwise, create a new one.
            if vector_store:
                logging.info("Updating existing vector store with new documents.")
                vector_store.add_documents(docs)
            else:
                logging.info("Creating new vector store for user.")
                vector_store = FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

            # Save the updated vector store to disk
            vector_store.save_local(f'document_embeddings/{user.id}')

            return Response({
                "message": "Document uploaded and vector store updated successfully."
            }, status=status.HTTP_200_OK)

        except:
            pass
        # finally:
        #     # Cleanup the uploaded file from the temporary storage
        #     if os.path.exists(file_path):
        #         os.remove(file_path)

    # GET: Handle question answering using the existing vector store
    def get(self, request):
        user = request.user
        vector_store_path = f'document_embeddings/{user.id}'

        # Check if the user's vector store exists
        if not os.path.exists(vector_store_path):
            return Response({"error": "Please upload a document first."}, status=status.HTTP_404_NOT_FOUND)

        try:
            # Load the existing vector store
            vector_store = FAISS.load_local(vector_store_path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
            return Response({"error": "Failed to load docs"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Extract the question from query parameters
        question = request.data.get('question')
        if not question:
            return Response({"error": "No question provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Set up the RAG chain with the vector store
        rag_chain = setup_rag_chain(vector_store)

        start_time = time.time()
        result = rag_chain.invoke({"query": question})
        end_time = time.time()

        return Response({
            "answer": result['result'],
            "time_taken": end_time - start_time
        }, status=status.HTTP_200_OK)
                
    def delete(self, request):
        document_name = request.data.get('document_name')
        if not document_name:
            return Response({"error": "No document name provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Construct the file path (assuming all documents are saved in 'tmp/')
        file_path = f'tmp/{document_name}'
        user = request.user

        try:
            # Retrieve the PDFDocument from the database
            pdf_doc = PDFDocument.objects.get(file_path=file_path, user=user)
        except PDFDocument.DoesNotExist:
            return Response({"error": "Document not found or not owned by the user"}, status=status.HTTP_404_NOT_FOUND)

        # Delete the PDF file from storage
        if os.path.exists(pdf_doc.file_path):
            logging.info(f"Deleting file: {pdf_doc.file_path}")
            default_storage.delete(pdf_doc.file_path)
        else:
            return Response({"error": "File does not exist on disk"}, status=status.HTTP_404_NOT_FOUND)

        # Load the user's vector store
        vector_store_path = f'document_embeddings/{user.id}'
        if not os.path.exists(vector_store_path):
            return Response({"error": "No vector store found for the user"}, status=status.HTTP_404_NOT_FOUND)

        try:
            vector_store = FAISS.load_local(vector_store_path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
            return Response({"error": "Failed to load vector store"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # You would need to implement logic to remove the document's embeddings from the vector store.
        # Unfortunately, FAISS itself does not have a built-in method to remove specific vectors by ID.
        # However, you can rebuild the vector store excluding the specific document.

        # Delete the PDF record from the database
        pdf_doc.delete()

        # Optionally, save the updated vector store
        vector_store.save_local(vector_store_path)

        return Response({"message": "Document and associated data deleted successfully"}, status=status.HTTP_200_OK)
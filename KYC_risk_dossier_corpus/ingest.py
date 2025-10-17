# ingest.py
import os
import tempfile
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from google.cloud import storage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from typing import List, Dict, Any
from langchain_core.documents import Document # Ensure Document is imported
import re
import logging
 
# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
 
if os.path.exists(".env"):
    load_dotenv()
else:
    logging.warning("Warning: .env file not found. Ensure environment variables are set.")
 
# Database credentials
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")
 
# GCS bucket name
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
 
# Gemini API Configuration
GEMINI_EMBEDDING_MODEL_NAME = "models/embedding-001" # Vertex AI embedding model
 
# Text splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
 
# Client information
CLIENT_NAME = "Apex Global Services FZE"
 
# --- PGVector Connection ---
# Corrected connection string to include port
PGVECTOR_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
COLLECTION_NAME = "risk_dossier_corpus"
 
# --- PDF Handling Functions ---
 
def extract_text_from_pdf_local(file_path: str) -> str:
    """Extracts text from a local PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except FileNotFoundError:
        logging.error(f"Error: Local file not found at {file_path}")
        return ""
    except Exception as e:
        logging.error(f"Error reading local PDF {file_path}: {e}")
        return ""
 
def extract_text_from_pdf_gcs(bucket_name: str, blob_name: str) -> str:
    """Extracts text from a PDF file stored in Google Cloud Storage."""
    if not bucket_name:
        logging.error("GCS bucket name is not configured. Cannot fetch PDFs from GCS.")
        return ""
 
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
 
        fd, local_path = tempfile.mkstemp()
        os.close(fd)
 
        logging.info(f"Downloading GCS file {blob_name} to temporary path: {local_path}")
        blob.download_to_filename(local_path)
 
        logging.info(f"Processing temporary file: {local_path}")
        text = extract_text_from_pdf_local(local_path) # This function calls extract_text_from_pdf_local
        return text
    except Exception as e:
        logging.error(f"Error downloading or processing GCS file {blob_name}: {e}")
        return ""
    finally:
        if 'local_path' in locals() and local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
                logging.info(f"Cleaned up temporary file: {local_path}")
            except Exception as cleanup_e:
                logging.error(f"Error during cleanup of {local_path}: {cleanup_e}")
 
# This function is the one that main.py will import and use.
def extract_text_from_pdf_gcs_single(bucket_name: str, blob_name: str) -> str:
    """Extracts text from a single PDF file stored in Google Cloud Storage."""
    return extract_text_from_pdf_gcs(bucket_name, blob_name) # Directly call the worker function
 
def get_pdf_paths_from_gcs(bucket_name: str) -> List[str]:
    """Lists all PDF files in the specified GCS bucket."""
    if not bucket_name:
        logging.error("GCS_BUCKET_NAME is not set in .env. Cannot fetch PDFs from GCS.")
        return []
 
    try:
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name)
        pdf_blobs = [blob.name for blob in blobs if blob.name.lower().endswith('.pdf')]
        logging.info(f"Found {len(pdf_blobs)} PDF files in GCS bucket {bucket_name}.")
        return pdf_blobs
    except Exception as e:
        logging.error(f"Error listing blobs in GCS bucket {bucket_name}: {e}")
        return []
 
# --- Embedding and Chunking Functions ---
def get_embeddings_model():
    """Initializes and returns the Gemini embeddings model using Vertex AI."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL_NAME
        )
        logging.info("Gemini embeddings model initialized successfully using ADC.")
        return embeddings
    except Exception as e:
        logging.error(f"Error initializing Gemini embeddings model: {e}")
        return None
 
def split_text_into_chunks(text: str) -> List[Document]: # Return List[Document] for PGVector
    """Splits text into manageable chunks and returns them as Langchain Documents."""
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(text)
    logging.info(f"Split text into {len(chunks)} chunks.")
    return [Document(page_content=chunk) for chunk in chunks]
 
# --- PGVector Insertion ---
def ingest_to_pgvector(documents: List[Document]):
    """Ingests documents into PGVector."""
    if not documents:
        logging.warning("No documents to ingest.")
        return
 
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
        logging.error("Database credentials are not fully set in .env. Cannot initialize PGVector.")
        return
 
    try:
        embeddings_model = get_embeddings_model()
        if not embeddings_model:
            logging.error("Failed to initialize embeddings model for PGVector. Aborting ingestion.")
            return
 
        logging.info(f"Initializing PGVector store for ingestion using connection string: {PGVECTOR_CONNECTION_STRING.split('@')[0]}@...(hidden credentials)")
        vector_store_ingest = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=PGVECTOR_CONNECTION_STRING,
            embedding_function=embeddings_model,
        )
 
        logging.info(f"Adding {len(documents)} documents to PGVector collection '{COLLECTION_NAME}'...")
        vector_store_ingest.add_documents(documents)
        logging.info(f"Successfully added {len(documents)} documents to PGVector.")
 
    except Exception as e:
        logging.error(f"Error during PGVector ingestion: {e}")
 
# --- Main Ingestion Logic ---
def extract_year_from_filename(filename: str) -> int | None:
    """
    Extracts the year from the filename, prioritizing formats like '...-YYYY.pdf'
    or '... YYYY ... .pdf'.
    """
    year_patterns = [
        r'\b(20\d{2})\b',  # Matches exactly 2000-2099 (e.g., 2021, 2023)
        r'review of (\d{4})', # e.g., "review of 2021"
        r'(\d{4}) review',  # e.g., "2023 review"
        r'in (\d{4})',      # e.g., "in 2022"
        r'(\d{2})review',   # e.g., "21review" (less common but covers some cases)
        r'(\d{2})[\s_-](\d{2})', # e.g., "20-21", "21-22" - try to capture the first year
        r'KYC Risk Dossier:.*?(\d{4})' # e.g., KYC Risk Dossier: 2020 Periodic Review
    ]
 
    found_year = None
 
    for pattern in year_patterns:
        matches = re.findall(pattern, filename, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple): # Handle patterns that capture multiple groups
                year_str = match[0] # Take the first captured group
            else:
                year_str = match
 
            try:
                year = int(year_str)
                if 2000 <= year <= 2030: # Plausible year range
                    if found_year is None or year < found_year: # Prioritize earlier years if multiple found
                         found_year = year
                         logging.info(f"Found potential year {year} from filename '{filename}' using pattern '{pattern}'.")
            except ValueError:
                logging.warning(f"Could not parse year from filename '{filename}' (matched '{match}').")
                continue
 
        if found_year is not None:
            # If you want to be strict and stop at the first valid year found, uncomment break.
            # break
            pass # Continue to check other patterns
 
    if found_year:
        logging.info(f"Final extracted year for filename '{filename}' is {found_year}.")
        return found_year
    else:
        logging.warning(f"No valid 4-digit year (2000-2030) found in the filename: '{filename}'.")
        return None
 
 
def ingest_pdfs_from_gcs():
    """Ingests PDFs from GCS and inserts them into the PGVector database."""
 
    gcs_pdf_files = get_pdf_paths_from_gcs(GCS_BUCKET_NAME)
 
    if not gcs_pdf_files:
        logging.warning("No PDF files found in the specified GCS bucket. Aborting GCS ingestion.")
        return
 
    all_documents_to_ingest = []
 
    for gcs_file_path in gcs_pdf_files:
        review_year = extract_year_from_filename(gcs_file_path)
        if review_year is None:
            logging.warning(f"Skipping file '{gcs_file_path}' due to invalid year format or extraction failure.")
            continue
 
        logging.info(f"\n--- Processing GCS file: '{gcs_file_path}' for year {review_year} ---")
        pdf_text = extract_text_from_pdf_gcs(GCS_BUCKET_NAME, gcs_file_path)
 
        if not pdf_text:
            logging.warning(f"Skipping file '{gcs_file_path}' due to no text extracted or an error occurred during processing.")
            continue
 
        text_chunks = split_text_into_chunks(pdf_text)
 
        if not text_chunks:
            logging.warning(f"No text chunks generated from '{gcs_file_path}'.")
            continue
 
        for chunk in text_chunks:
            chunk.metadata["source_file"] = gcs_file_path
            chunk.metadata["client_name"] = CLIENT_NAME
            chunk.metadata["review_year"] = review_year
            logging.debug(f"Assigned metadata to chunk: {chunk.metadata}")
 
        all_documents_to_ingest.extend(text_chunks)
        logging.info(f"Prepared {len(text_chunks)} chunks with metadata for year {review_year} from '{gcs_file_path}'.")
 
    if not all_documents_to_ingest:
        logging.warning("No documents were prepared for ingestion. Aborting.")
        return
 
    ingest_to_pgvector(all_documents_to_ingest)
 
    logging.info("GCS data ingestion process completed.")
 
if __name__ == "__main__":
    logging.info("Running ingest_data.py as a script...")
    ingest_pdfs_from_gcs()
    logging.info("ingest_data.py script finished.")
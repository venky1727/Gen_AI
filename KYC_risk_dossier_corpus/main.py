# main.py
import streamlit as st
import os
import re
import logging
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.cloud import storage
import tempfile
from PyPDF2 import PdfReader
from typing import List, Dict, Any
from langchain_core.documents import Document
 
# Import ALL necessary helper functions from ingest.py
try:
    from ingest import (
        extract_year_from_filename,
        split_text_into_chunks,
        ingest_pdfs_from_gcs,
        extract_text_from_pdf_local,
        extract_text_from_pdf_gcs_single
    )
except ImportError as e:
    st.error(f"Could not import helper functions from ingest.py: {e}. Make sure 'ingest.py' is in the same directory or accessible and correctly defines these functions.")
    st.stop()
 
# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
 
logging.info("Loading .env file...")
load_dotenv()
 
# Vertex AI Gemini Model Names
GEMINI_LLM_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "models/embedding-001"
 
# --- Client Information ---
CLIENT_NAME = "Apex Global Services FZE"
 
# --- PGVector Connection String Logic ---
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")
 
if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    st.error("Database credentials are not fully set in .env. Please check your .env file.")
    st.stop()
 
# Corrected connection string to include port
PGVECTOR_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
COLLECTION_NAME = "risk_dossier_corpus"
 
# --- Initialize Models ---
@st.cache_resource
def get_gemini_llm():
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_LLM_MODEL_NAME,
            temperature=0.3,
            # convert_system_message_to_human=True is deprecated. Forcing it to False or removing it.
            convert_system_message_to_human=False
        )
        logging.info("Gemini chat model initialized successfully using Vertex AI/ADC.")
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Gemini chat model: {e}")
        logging.error(f"Failed to initialize Gemini chat model: {e}")
        return None
 
@st.cache_resource
def get_gemini_embeddings():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME
        )
        logging.info("Gemini embeddings model initialized successfully using Vertex AI/ADC.")
        return embeddings
    except Exception as e:
        st.error(f"Failed to initialize Gemini embeddings model: {e}")
        logging.error(f"Failed to initialize Gemini embeddings model: {e}")
        return None
 
# --- PGVector Store Initialization ---
@st.cache_resource
def get_vector_store():
    try:
        embeddings = get_gemini_embeddings()
        if not embeddings:
            st.error("Embeddings model not available. Cannot initialize vector store.")
            return None
        logging.info(f"Initializing PGVector store with connection string: {PGVECTOR_CONNECTION_STRING.split('@')[0]}@...(hidden credentials)")
        vector_store = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=PGVECTOR_CONNECTION_STRING,
            embedding_function=embeddings,
        )
        logging.info("PGVector store initialized successfully.")
        return vector_store
    except Exception as e:
        st.error("Failed to initialize PGVector store.")
        st.error(f"Connection String Used: {PGVECTOR_CONNECTION_STRING.split('@')[0]}@...(hidden credentials)")
        st.error(f"Error: {e}")
        st.error("Please ensure the database is accessible, credentials are correct, and the collection/table exists with the correct schema and index.")
        return None
 
# --- Helper to extract year from query (More Robust) ---
def extract_year_from_query(query: str) -> int | None:
    """
    Extracts a 4-digit year from the query, prioritizing phrases like '2023 review'
    or 'in 2022'.
    """
    year_patterns = [
        # More specific patterns first
        r'\b(20\d{2})\b',  # Matches exactly 2000-2099 (e.g., 2021, 2023)
        r'review of (\d{4})', # e.g., "review of 2021"
        r'(\d{4}) review',  # e.g., "2023 review"
        r'in (\d{4})',      # e.g., "in 2022"
        r'(\d{2})review',   # e.g., "21review" (less common but covers some cases)
        r'(\d{2})-(\d{2})', # e.g., "2021-2022" - try to capture the first year
        r'KYC Risk Dossier:.*?(\d{4})' # e.g., KYC Risk Dossier: 2020-2024 Periodic Review
    ]
 
    found_year = None
 
    for pattern in year_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple): # Handle patterns that capture multiple groups (like (\d{2})-(\d{2}))
                year_str = match[0] # Take the first captured group
            else: # Handle single group or direct match
                year_str = match
 
            try:
                year = int(year_str)
                # Basic validation for plausible years (adjust range as needed)
                if 2000 <= year <= 2030:
                    if found_year is None or year < found_year: # Prioritize earlier years if multiple found
                         found_year = year
                         logging.info(f"Found potential year {year} from query '{query}' using pattern '{pattern}'.")
            except ValueError:
                logging.warning(f"Could not parse year from query '{query}' (matched '{match}').")
                continue
 
        if found_year is not None:
            # If you want to be strict and stop at the first valid year found, uncomment break.
            # break
            pass # Continue to check other patterns to potentially find a better year
 
    if found_year:
        logging.info(f"Final extracted year for query '{query}' is {found_year}.")
        return found_year
    else:
        logging.warning(f"No valid 4-digit year (2000-2030) found in the query: '{query}'.")
        return None
 
# --- RAG Chain Setup with UI Enhancements ---
def setup_rag_chain(vector_store, llm):
    if not vector_store or not llm:
        logging.error("Vector store or LLM is not initialized. Cannot set up RAG chain.")
        return None
 
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": 30} # Fetch top 10 similar documents
    )
 
    template = """
    You are an AI assistant specializing in summarizing Know Your Customer (KYC) risk assessment dossiers.
    Your task is to provide a concise summary of the client's risk profile for a specific year,
    and then state the identified risk level.
 
    Use the following pieces of context (relevant chunks from the risk dossier) to answer the question.
    Focus on information pertaining to the specified year if a year is mentioned in the question.
 
    **Instructions:**
    1.  Provide a "Summary:" section with 3-4 sentences covering key findings and risk drivers for the specified year.
        **If no specific year was provided in the query, state 'Summary for unspecified year:'**
    2.  Provide a "Risk Level:" section with the identified risk level (e.g., High, Medium, Low, Critical).
    3.  If no relevant information is found for the specified year or the query, state "No information found for this query." in both sections.
    4.  Do NOT include any conversational filler or introductory phrases.
 
    Context:
    {context}
 
    Question:
    {question}
 
    Helpful Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
 
    rag_chain_core = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
 
    def invoke_rag_with_filtered_retrieval(query_input):
        logging.info(f"--- Processing Query: '{query_input}' ---")
        query_year = extract_year_from_query(query_input) # Use the improved function
        logging.info(f"Extracted year from query: {query_year}")
 
        retrieved_docs = []
        all_relevant_docs = []
 
        try:
            # Construct the filter for the retriever based on the query_year
            retriever_filter = None
            if query_year is not None:
                # The filter for PGVector in Langchain expects a dictionary
                # where keys are metadata keys and values are the filter criteria.
                retriever_filter = {"review_year": query_year}
                logging.info(f"Applying retriever filter: {retriever_filter}")
           
            # Pass the filter to the retriever
            # This is the CRITICAL change: applying the filter directly to the retriever
            all_relevant_docs = base_retriever.invoke(query_input, filter=retriever_filter)
           
            logging.info(f"Retrieved {len(all_relevant_docs)} documents after applying filter.")
           
            context_for_llm = "No relevant context found for your query." # Default message
 
            if all_relevant_docs:
                # If docs were found after filtering, format them.
                # The filter should have already ensured they are for the correct year.
                context_for_llm = format_docs_with_year_filter(all_relevant_docs, query_year)
            else:
                # If no docs found after filtering, provide a specific message.
                if query_year is not None:
                    logging.warning(f"No documents found matching the query year {query_year} after retrieval and filtering.")
                    context_for_llm = f"No relevant context found for the year {query_year}."
                else:
                    # This case should ideally not happen if all_relevant_docs is empty and query_year is None,
                    # but for safety:
                    context_for_llm = "No relevant context found for your query."
 
            logging.debug(f"Context prepared for LLM (first 500 chars):\n{context_for_llm[:500]}...")
 
        except Exception as e:
            logging.error(f"Error during retrieval or filtering for query '{query_input}': {e}")
            context_for_llm = f"An error occurred during document retrieval: {e}"
       
        final_prompt_input = {
            "context": context_for_llm,
            "question": query_input
        }
 
        try:
            response = rag_chain_core.invoke(final_prompt_input)
            logging.info(f"LLM response generated successfully.")
            return response
        except Exception as e:
            logging.error(f"Error during RAG chain invocation for query '{query_input}': {e}")
            return f"An error occurred while processing your request: {e}"
 
    # This function remains the same, as the filtering is now done by the retriever.
    # It's still useful for ensuring the output context is clean if any relevant docs are passed.
    def format_docs_with_year_filter(docs, year):
        """Formats documents, only including those that match the specified year."""
        if not docs:
            logging.warning("No documents provided to format.")
            return "No relevant context found for the query."
 
        if year is not None:
            # This filter is now redundant if the retriever already filtered,
            # but it's good for safety and ensures the output is clean.
            relevant_docs = [doc for doc in docs if doc.metadata.get('review_year') == year]
            logging.debug(f"Formatting: Found {len(relevant_docs)} docs for year {year}.")
            if not relevant_docs:
                logging.warning(f"Formatting: No documents found matching year {year} in the provided list.")
                return "No relevant context found for the specified year."
            else:
                docs_to_format = relevant_docs
        else:
            docs_to_format = docs
            logging.debug(f"Formatting: Using all {len(docs_to_format)} documents (no year specified).")
 
        # Sort by source file for consistent output
        docs_to_format.sort(key=lambda d: d.metadata.get('source_file', ''))
       
        formatted_text = ""
        current_source = None
        for doc in docs_to_format:
            source = doc.metadata.get('source_file', 'Unknown Source')
            if source != current_source:
                if current_source is not None:
                    formatted_text += "\n\n"
                formatted_text += f"--- Source: {source} ---\n"
                current_source = source
            formatted_text += doc.page_content + "\n"
       
        logging.debug(f"Formatted context length: {len(formatted_text)}")
        return formatted_text.strip()
 
    base_rag_chain = invoke_rag_with_filtered_retrieval
   
    logging.info("RAG chain setup complete.")
    return base_rag_chain
 
# --- Streamlit UI Functions ---
 
def get_gcs_bucket_name():
    return os.getenv("GCS_BUCKET_NAME")
 
def process_user_source(source_type: str, source_value: str, year: int | None = None):
    """Processes a single file from GCS or local path and returns Langchain Documents."""
    all_docs = []
    if not source_value:
        st.warning("No file source provided.")
        return all_docs
 
    try:
        pdf_text = ""
        source_file_name = ""
        file_year = year
 
        if source_type == "GCS Path":
            bucket_name = get_gcs_bucket_name()
            if not bucket_name:
                st.error("GCS bucket name is not configured in .env.")
                return all_docs
            if not source_value.lower().endswith('.pdf'):
                st.error("GCS path must point to a PDF file.")
                return all_docs
 
            source_file_name = source_value
            # Use the CORRECTLY IMPORTED GCS function
            pdf_text = extract_text_from_pdf_gcs_single(bucket_name, source_value)
            if file_year is None:
                file_year = extract_year_from_filename(source_file_name)
                if file_year is None:
                    logging.warning(f"Could not determine year for GCS file '{source_file_name}'. Proceeding without year metadata for this file.")
 
        elif source_type == "Local File Path":
            if not source_value.lower().endswith('.pdf'):
                st.error("Local file path must point to a PDF file.")
                return all_docs
            source_file_name = os.path.basename(source_value)
            # Use the CORRECTLY IMPORTED local function
            pdf_text = extract_text_from_pdf_local(source_value)
            if file_year is None:
                file_year = extract_year_from_filename(source_file_name)
                if file_year is None:
                    logging.warning(f"Could not determine year for local file '{source_file_name}'. Proceeding without year metadata for this file.")
 
        if not pdf_text:
            st.warning(f"Could not extract text from '{source_value}'. Skipping.")
            return all_docs
 
        logging.info(f"Processing file: '{source_file_name}' for year {file_year if file_year is not None else 'Unknown'}")
 
        # Use the CORRECTLY IMPORTED chunking function
        text_chunks = split_text_into_chunks(pdf_text)
        if not text_chunks:
            st.warning(f"No text chunks generated from '{source_file_name}'.")
            return all_docs
 
        for chunk in text_chunks:
            chunk.metadata["source_file"] = source_file_name
            chunk.metadata["client_name"] = CLIENT_NAME
            # Ensure metadata['review_year'] is consistently stored as an integer
            chunk.metadata["review_year"] = file_year if isinstance(file_year, int) else int(file_year) if isinstance(file_year, str) and file_year.isdigit() else -1
            logging.debug(f"Assigned metadata to chunk: {chunk.metadata}")
 
        all_docs.extend(text_chunks)
        logging.info(f"Prepared {len(text_chunks)} chunks with metadata from '{source_file_name}'.")
        return all_docs
 
    except Exception as e:
        logging.error(f"An error occurred processing source '{source_value}': {e}")
        return all_docs
 
# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(
        page_title="KYC Risk Dossier Analyzer",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📊"
    )
 
    with st.sidebar:
        st.title("Configuration")
        st.markdown("---")
 
        source_option = st.radio(
            "Choose data source:",
            ("Use Predefined GCS Files", "Enter Specific File Paths"),
            key="source_option_radio"
        )
 
        # Use session state to manage uploaded documents that are pending ingestion
        if 'uploaded_docs' not in st.session_state:
            st.session_state['uploaded_docs'] = []
 
        if source_option == "Use Predefined GCS Files":
            st.info("Using PDFs from the GCS bucket defined in `.env`.")
            st.caption("Ensure `ingest_data.py` has been run recently to populate the database.")
            if st.button("Reload Data from GCS", key="reload_gcs"):
                with st.spinner("Ingesting PDFs from GCS..."):
                    logging.info("Initiating GCS data ingestion...")
                    ingest_pdfs_from_gcs() # Call the imported function
                    st.success("GCS data ingestion process completed. Please refresh the page or re-enter your query.")
 
        elif source_option == "Enter Specific File Paths":
            st.warning("Adding files here will attempt to ingest them into the database.")
           
            gcs_path = st.text_input("GCS File Path (e.g., folder/your_doc.pdf)", key="gcs_input")
            if gcs_path:
                default_year_gcs = extract_year_from_filename(gcs_path) or 2020-2024
                gcs_year = st.number_input("Year for GCS file (optional, if not in path)", min_value=2000, max_value=2030, value=default_year_gcs, key="gcs_year_input", label_visibility="collapsed")
 
                if st.button("Add GCS File to Session", key="add_gcs"):
                    with st.spinner("Processing GCS file..."):
                        docs = process_user_source("GCS Path", gcs_path, gcs_year) # Use imported function
                        if docs:
                            st.session_state['uploaded_docs'].append((gcs_path, docs))
                            st.success(f"Added {len(docs)} chunks from GCS: {os.path.basename(gcs_path)}")
                        else:
                            st.error(f"Failed to process GCS file: {gcs_path}")
 
            st.markdown("---")
            local_file = st.file_uploader("Upload Local PDF File", type=["pdf"], key="local_upload")
            if local_file:
                default_year_local = extract_year_from_filename(local_file.name) or 2020-2024
                local_year = st.number_input("Year for local file (optional, if not in filename)", min_value=2000, max_value=2030, value=default_year_local, key="local_year_input", label_visibility="collapsed")
 
                if st.button("Add Local File to Session", key="add_local"):
                    with st.spinner("Processing local file..."):
                        # Process the uploaded file directly without saving to temp file immediately for adding to session
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="streamlit_upload_") as tmp_file:
                            tmp_file.write(local_file.getvalue())
                            local_file_path = tmp_file.name
                       
                        docs = process_user_source("Local File Path", local_file_path, local_year) # Use imported function
                        if docs:
                            st.session_state['uploaded_docs'].append((local_file.name, docs))
                            st.success(f"Added {len(docs)} chunks from local file: {local_file.name}")
                        else:
                            st.error(f"Failed to process local file: {local_file.name}")
                        # Clean up the temp file after processing
                        if os.path.exists(local_file_path):
                            os.remove(local_file_path)
 
            st.markdown("---")
            if st.session_state['uploaded_docs']:
                st.write("Files ready for ingestion:")
                for filename, _ in st.session_state['uploaded_docs']:
                    st.write(f"- {filename}")
 
                if st.button("Ingest Selected Files to DB", key="ingest_specific"):
                    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
                        st.error("Database credentials not configured. Cannot ingest.")
                        return
                   
                    all_docs_to_ingest_batch = []
                    for _, doc_list in st.session_state['uploaded_docs']:
                        all_docs_to_ingest_batch.extend(doc_list)
 
                    if not all_docs_to_ingest_batch:
                        st.warning("No files selected for ingestion.")
                        return
 
                    try:
                        embeddings_model = get_gemini_embeddings()
                        if not embeddings_model:
                            st.error("Embeddings model not initialized. Cannot ingest.")
                            return
 
                        logging.info(f"Initializing PGVector store for ingestion: {PGVECTOR_CONNECTION_STRING.split('@')[0]}@...(hidden credentials)")
                        vector_store_ingest = PGVector(
                            collection_name=COLLECTION_NAME,
                            connection_string=PGVECTOR_CONNECTION_STRING,
                            embedding_function=embeddings_model,
                        )
                        logging.info(f"Adding {len(all_docs_to_ingest_batch)} documents to PGVector collection '{COLLECTION_NAME}'...")
                        vector_store_ingest.add_documents(all_docs_to_ingest_batch)
                        st.success(f"Successfully ingested {len(all_docs_to_ingest_batch)} documents into the database.")
                        st.balloons()
                        st.session_state['uploaded_docs'].clear() # Clear after ingestion
                        st.rerun() # Rerun to refresh the UI state
                    except Exception as e:
                        logging.error(f"Error during specific file ingestion: {e}")
                        st.error(f"Error during ingestion: {e}")
            elif not st.session_state['uploaded_docs']:
                st.info("Add GCS files or upload local files first.")
 
        st.markdown("---")
        st.write("Version: 1.0.8") # Version update
 
    st.title(f"KYC Risk Dossier Analyzer")
    st.markdown(f"**Client:** {CLIENT_NAME}")
    st.markdown("---")
 
    llm = get_gemini_llm()
    vector_store = get_vector_store()
 
    if not llm or not vector_store:
        st.error("Could not initialize core components. Please check logs and configurations.")
        st.stop()
 
    rag_chain = setup_rag_chain(vector_store, llm)
 
    if not rag_chain:
        st.error("Failed to set up the RAG chain. Please check logs.")
        st.stop()
 
    st.subheader("Enter your query:")
    query = st.text_input(
        "Ask a question about the risk dossier (e.g., 'Summarize the 2021 review', 'What was the risk rating in 2022?', 'What is the current risk profile?'):",
        placeholder="Type your question here...",
        label_visibility="collapsed"
    )
 
    if query:
        with st.spinner("Analyzing and generating response..."):
            ai_response = rag_chain(query)
 
            summary_section = "No summary available."
            risk_level_section = "No risk level found."
 
            # Adjusted regex to better capture sections, ensuring it handles cases where one might be missing.
            # Make sure the summary section ends before "Risk Level:" or the end of the string.
            summary_match = re.search(r"Summary:(.*?)(?:Risk Level:|$)", ai_response, re.DOTALL | re.IGNORECASE)
            # Make sure the risk level section is captured correctly.
            risk_match = re.search(r"Risk Level:(.*?)(?:Summary:|$)", ai_response, re.DOTALL | re.IGNORECASE)
 
 
            if summary_match and "No information found for this query." not in summary_match.group(1):
                summary_section = summary_match.group(1).strip()
            elif "No information found for this query." in ai_response:
                summary_section = "No information found for this query."
            else:
                summary_section = "No summary found in the response."
 
            if risk_match and "No information found for this query." not in risk_match.group(1):
                risk_level_section = risk_match.group(1).strip()
            elif "No information found for this query." in ai_response:
                risk_level_section = "No information found for this query."
            else:
                risk_level_section = "No risk level found in the response."
 
            col1, col2 = st.columns(2)
 
            with col1:
                st.subheader("Summary:")
                # MODIFIED THIS LINE: Removed the code block formatting ```markdown\n...\n```
                # This will render the markdown directly, allowing it to expand.
                st.markdown(summary_section)
 
 
            with col2:
                st.subheader("Risk Level:")
                risk_text = risk_level_section.strip()
                if "high" in risk_text.lower() or "critical" in risk_text.lower():
                    st.error(f"**{risk_text}**")
                elif "medium" in risk_text.lower():
                    st.warning(f"**{risk_text}**")
                elif "low" in risk_text.lower():
                    st.success(f"**{risk_text}**")
                else:
                    st.info(f"**{risk_text}**")
    else:
        st.info("Please enter a question or select a file source from the sidebar to begin.")
        st.subheader("Default View (Information on 2020-2024 Review):")
        st.markdown("""
        The KYC Risk Dossier for Apex Global Services FZE shows a progression of risk assessments over the years.
        In the **2020-2024 Periodic Review**, the client was assessed as **Medium Risk**. Key drivers included its UAE domicile and use of Trade Finance products. These were mitigated by strong parentage (Orion Foundation), a plausible business model, low/consistent activity, and transparency. The recommendation was **APPROVAL** with a two-year review cycle.
 
        *Note: Enter a specific year in your query (e.g., 'Summarize the 2021 review') to get year-specific insights.*
        """)
 
if __name__ == "__main__":
    main()
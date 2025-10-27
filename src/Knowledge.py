# knowledge.py - Document processing and vector store builder for SusTech Recycling Agent
import os
import sys
import logging
from datetime import datetime
from typing import List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
import time

# Add the src directory to the Python path so we can import from utils and agent modules
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.vectorstores import FAISS

from utils.cleaner import clean_documents

# Load environment variables from .env file
load_dotenv()

# Configure logging
current_script_dir = os.path.dirname(os.path.abspath(__file__))
log_filename = os.path.join(current_script_dir, ".logs", f"cleaning_{datetime.now().strftime('%Y%m%d')}.log")
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode="a", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the directory containing the recycling documents
DOCS_DIR = os.path.join(os.path.dirname(__file__), "recycling_docs")

# Define supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".txt", ".md"}

# Initialize the text splitter (more efficient configuration)
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")], strip_headers=False)

# Performance settings
MAX_WORKERS = min(multiprocessing.cpu_count(), 4)
BATCH_SIZE = 50
CHUNK_SIZE = 800

# NEW: Test run flag
TEST_RUN = os.getenv("TEST_RUN", "False").lower() == "true"
TEST_FILE_NAME = os.getenv("TEST_FILE_NAME", None)  # Optional: specific file for test run

r
def infer_region_from_filename(filename: str) -> str:
    """
    Infer the region (US or Germany) from the filename.
    """
    filename_lower = filename.lower()
    if "germany" in filename_lower or "berlin" in filename_lower:
        return "Germany"
    elif "us" in filename_lower or "california" in filename_lower or "calrecycle" in filename_lower or "america" in filename_lower:
        return "US"
    else:
        # Default to Germany for now, or could be "General"
        return "Germany"


def get_embedding_model():
    """Get embedding model with caching to avoid re-initialization."""
    if not hasattr(get_embedding_model, "_cached_model"):
        if os.getenv("GEMINI_API_KEY"):
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            print("🚀 Using Google Generative AI Embeddings...")
            get_embedding_model._cached_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GEMINI_API_KEY"))
        else:
            raise ValueError("No valid embedding API credentials found. Please check your .env file.")

    return get_embedding_model._cached_model


def load_and_clean_file_fast(filepath: str) -> List[Document]:
    """
    Optimized file loading with correct page number preservation.
    """
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)
    region = infer_region_from_filename(filename)

    try:
        if ext == ".pdf":
            # Use page mode to preserve individual page information
            loader = PyMuPDF4LLMLoader(file_path=filepath, mode="page")
            docs = loader.load()

            # Ensure page numbers are correctly set in metadata
            for i, doc in enumerate(docs):
                if "page" not in doc.metadata or doc.metadata["page"] is None:
                    # Extract page number from source if available, otherwise use index + 1
                    doc.metadata["page"] = i + 1

        elif ext == ".epub":
            loader = UnstructuredEPubLoader(filepath, mode="elements")
            docs = loader.load()
        elif ext in [".txt", ".md", ".html"]:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            docs = [Document(page_content=content, metadata={"source": filepath, "filename": filename, "page": 1})]
        else:
            return []

        # Streamlined cleaning with less aggressive filtering
        if docs:
            cleaned_docs, _ = clean_documents(docs, min_content_length=50, verbose=False)
            # Add filename and ensure page numbers are preserved
            for doc in cleaned_docs:
                doc.metadata["filename"] = filename
                doc.metadata["file_type"] = ext
                doc.metadata["region"] = region
                # Ensure page number exists
                if "page" not in doc.metadata:
                    doc.metadata["page"] = 1
            return cleaned_docs
        return []

    except Exception as e:
        logging.error(f"Failed to process {filepath}: {e}")
        return []


def chunk_documents_fast(docs: List[Document]) -> List[Document]:
    """
    Optimized chunking with page number preservation.
    """
    chunks = []
    for doc in docs:
        # Use a more efficient text splitter for speed
        split_chunks = text_splitter.split_text(doc.page_content)
        for chunk in split_chunks:
            # Limit chunk size for faster embedding
            if len(chunk.page_content) > CHUNK_SIZE:
                # Simple truncation for speed
                chunk.page_content = chunk.page_content[:CHUNK_SIZE] + "..."

            # Preserve all metadata including page numbers
            chunk.metadata.update(doc.metadata)
            chunks.append(chunk)

    return chunks


def process_file_batch(file_batch: List[str]) -> List[Document]:
    """
    Process a batch of files in parallel using threads.
    """
    all_chunks = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all file loading tasks
        future_to_file = {executor.submit(load_and_clean_file_fast, filepath): filepath for filepath in file_batch}

        # Collect results
        for future in as_completed(future_to_file):
            filepath = future_to_file[future]
            try:
                docs = future.result()
                if docs:
                    chunks = chunk_documents_fast(docs)
                    all_chunks.extend(chunks)
                    # Show page count information
                    page_info = f"pages {min(d.metadata.get('page', 1) for d in docs)}-{max(d.metadata.get('page', 1) for d in docs)}" if len(docs) > 1 else f"page {docs[0].metadata.get('page', 1)}"
                    print(f"✅ Processed {os.path.basename(filepath)}: {len(chunks)} chunks from {page_info}")
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(filepath)}: {e}")

    return all_chunks


def build_vectorstore_incrementally(all_chunks: List[Document], embedding_model, vector_store_dir: str):
    """
    Build vector store incrementally with batching for better memory management.
    """
    vector_store_path = os.path.join(vector_store_dir, "recycling_kb_vectorstore")
    os.makedirs(vector_store_dir, exist_ok=True)

    # Check if vector store exists
    vector_store = None
    if os.path.exists(vector_store_path):
        try:
            print("📦 Loading existing vector store...")
            vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
            print(f"✅ Loaded existing vector store with {vector_store.index.ntotal} documents")
        except Exception as e:
            print(f"⚠️  Failed to load existing vector store: {e}")
            vector_store = None

    # Process chunks in batches
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        try:
            if vector_store is None:
                # Create new vector store with first batch
                vector_store = FAISS.from_documents(batch, embedding_model)
                print(f"✅ Created new vector store")
            else:
                # Add batch to existing vector store
                vector_store.add_documents(batch)
                print(f"✅ Added batch to vector store")

            # Save after each batch to prevent data loss
            vector_store.save_local(vector_store_path)

        except Exception as e:
            print(f"❌ Error processing batch {batch_num}: {e}")
            continue

    return vector_store


def get_file_batches(doc_files: List[str], batch_size: int = 3) -> List[List[str]]:
    """
    Split files into batches for parallel processing.
    """
    batches = []
    for i in range(0, len(doc_files), batch_size):
        batches.append(doc_files[i : i + batch_size])
    return batches


def validate_page_numbers(chunks: List[Document]) -> None:
    """
    Validate that page numbers are correctly assigned to chunks.
    """
    print("\n📊 Page number validation:")
    page_counts = {}
    for chunk in chunks:
        page = chunk.metadata.get("page", "Unknown")
        filename = chunk.metadata.get("filename", "Unknown")
        key = f"{filename}_page_{page}"
        page_counts[key] = page_counts.get(key, 0) + 1

    # Show sample page distribution
    for key, count in list(page_counts.items())[:10]:  # Show first 10
        print(f"  {key}: {count} chunks")

    if len(page_counts) > 10:
        print(f"  ... and {len(page_counts) - 10} more page groups")


def main():
    """
    Optimized main function with parallel processing and batching.
    """
    start_time = time.time()

    print("🚀 SusTech Recycling Agent Knowledge Base Builder")
    print("=" * 60)

    if not os.path.isdir(DOCS_DIR):
        logging.error(f"Documents directory does not exist: {DOCS_DIR}")
        print(f"❌ Documents directory does not exist: {DOCS_DIR}")
        print("Please create the 'recycling_docs' directory and add your recycling law documents (PDF, EPUB, TXT, MD files).")
        return

    # Initialize embedding model
    try:
        embedding_model = get_embedding_model()
        print(f"✅ Embedding model initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize embedding model: {e}")
        return

    # Set up vector store directory
    vector_store_dir = os.path.join(os.path.dirname(__file__), "vector_store")

    doc_files = [os.path.join(DOCS_DIR, f) for f in os.listdir(DOCS_DIR) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]

    if not doc_files:
        logging.info("No supported document files found to process.")
        print("⚠️  No supported document files found to process.")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        print(f"Please add recycling law documents to the '{DOCS_DIR}' directory.")
        return

    # NEW: Apply test run logic
    if TEST_RUN:
        if TEST_FILE_NAME:
            # Try to find the specified test file
            found_test_file = False
            for f_path in doc_files:
                if os.path.basename(f_path) == TEST_FILE_NAME:
                    doc_files = [f_path]
                    found_test_file = True
                    break
            if not found_test_file:
                print(f"❌ Specified TEST_FILE_NAME '{TEST_FILE_NAME}' not found in {DOCS_DIR}. Exiting.")
                return
        else:
            # If no specific file is named, just take the first one
            doc_files = [doc_files[0]]
        print(f"📚 Processing {len(doc_files)} file(s) in test mode...")
    else:
        print(f"📚 Found {len(doc_files)} files to process...")

    print(f"💾 Vector store location: {vector_store_dir}")
    print()

    # Split files into batches for parallel processing
    # In test mode, we'll still use batches, but it will likely be a single batch
    file_batches = get_file_batches(doc_files, batch_size=6)
    all_chunks = []

    print(f"🔄 Processing {len(file_batches)} batches in parallel...")

    # Process file batches with progress bar
    with tqdm(total=len(doc_files), desc="Processing files") as pbar:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit batch processing tasks
            future_to_batch = {executor.submit(process_file_batch, batch): batch for batch in file_batches}

            # Collect results
            for future in as_completed(future_to_batch):
                try:
                    batch_chunks = future.result()
                    all_chunks.extend(batch_chunks)
                    pbar.update(len(future_to_batch[future]))
                except Exception as e:
                    print(f"❌ Error processing batch: {e}")
                    pbar.update(len(future_to_batch[future]))

    if not all_chunks:
        print("❌ No chunks were generated from any files")
        return

    print(f"\n📊 Generated {len(all_chunks)} total chunks")

    # Validate page numbers
    validate_page_numbers(all_chunks)

    print("🏗️  Building vector store...")

    # Build vector store incrementally
    try:
        vector_store = build_vectorstore_incrementally(all_chunks, embedding_model, vector_store_dir)

        if vector_store:
            end_time = time.time()
            processing_time = end_time - start_time

            print("\n🎉 Processing complete!")
            print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
            print(f"📄 Logs saved to: {log_filename}")
            print(f"💾 Vector store saved to: {os.path.join(vector_store_dir, 'recycling_kb_vectorstore')}")
            print(f"📊 Total documents in vector store: {vector_store.index.ntotal}")
            print(f"🚀 Processing speed: {len(all_chunks)/processing_time:.1f} chunks/second")
        else:
            print("❌ Failed to create vector store")

    except Exception as e:
        print(f"❌ Error building vector store: {e}")


if __name__ == "__main__":
    main()

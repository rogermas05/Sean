#!/usr/bin/env python3
"""
Large PDF Processor - RAG Pipeline for Very Large Documents

This script processes extremely large PDFs (1000+ pages) and creates a
vector database that can be used for retrieval-augmented generation (RAG).
"""

import os
import argparse
import time
from typing import List, Iterator
from tqdm import tqdm

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

def extract_text_by_page(pdf_path: str) -> Iterator[str]:
    """Extract text from PDF one page at a time to manage memory."""
    print(f"Extracting text from PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")
    
    for i in tqdm(range(total_pages), desc="Extracting pages"):
        try:
            page = reader.pages[i]
            yield page.extract_text() or ""
        except Exception as e:
            print(f"Error on page {i}: {e}")
            yield ""  # Return empty string on error

def process_large_pdf(
    pdf_path: str, 
    db_path: str,
    batch_size: int = 100, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200,
    force_rebuild: bool = False
) -> None:
    """
    Process a large PDF in batches to manage memory usage.
    
    Args:
        pdf_path: Path to the PDF file
        db_path: Path where to store the vector database
        batch_size: Number of pages to process in each batch
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        force_rebuild: Whether to force rebuilding the vector store
    """
    if os.path.exists(db_path) and not force_rebuild:
        print(f"Vector store already exists at {db_path}. Use --force to rebuild.")
        return
    
    # Create/recreate the directory if forcing rebuild
    if force_rebuild and os.path.exists(db_path):
        import shutil
        print(f"Removing existing database at {db_path}")
        shutil.rmtree(db_path)
    
    # Initialize embeddings and text splitter
    start_time = time.time()
    embeddings = OpenAIEmbeddings()
    vectorstore = None
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    # Process the PDF in batches
    pages_text = []
    page_count = 0
    total_chunks = 0
    
    for page_text in extract_text_by_page(pdf_path):
        pages_text.append(page_text)
        page_count += 1
        
        # Process a batch when we reach the batch size
        if page_count % batch_size == 0:
            chunks, temp_vectorstore = process_batch(
                pages_text,
                text_splitter,
                embeddings,
                vectorstore,
                db_path
            )
            
            total_chunks += chunks
            vectorstore = temp_vectorstore
            pages_text = []  # Clear the batch
    
    # Process any remaining pages
    if pages_text:
        chunks, vectorstore = process_batch(
            pages_text,
            text_splitter,
            embeddings,
            vectorstore,
            db_path
        )
        total_chunks += chunks
    
    elapsed_time = time.time() - start_time
    print(f"Completed processing {page_count} pages with {total_chunks} total chunks")
    print(f"Database saved to {db_path}")
    print(f"Total processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

def process_batch(
    pages_text: List[str],
    text_splitter: RecursiveCharacterTextSplitter,
    embeddings: OpenAIEmbeddings,
    vectorstore: Chroma,
    persist_directory: str
) -> tuple:
    """Process a batch of pages."""
    print(f"Processing batch of {len(pages_text)} pages...")
    
    # Join the pages and split into chunks
    text = "\n".join(pages_text)
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} chunks from batch")
    
    # If this is our first batch, create the vector store
    if vectorstore is None:
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    # Otherwise add to the existing vector store
    else:
        vectorstore.add_texts(chunks)
    
    print(f"Batch processed and saved to {persist_directory}")
    return len(chunks), vectorstore

def main():
    parser = argparse.ArgumentParser(
        description="Process Very Large PDFs (1000+ pages) for RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--db", default="./pdf_vectordb", help="Directory to store the vector database")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of pages to process in each batch")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks") 
    parser.add_argument("--force", action="store_true", help="Force rebuild of vector store")
    args = parser.parse_args()
    
    # Check if OPENAI_API_KEY is set
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it with: export OPENAI_API_KEY=your_api_key")
        return
    
    process_large_pdf(
        args.pdf, 
        args.db,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_rebuild=args.force
    )
    
    print("\nTo query this database, run:")
    print(f"python rag_query.py --db {args.db} --question \"Your question here\"")
    print("Or for interactive mode:")
    print(f"python rag_query.py --db {args.db}")

if __name__ == "__main__":
    main() 
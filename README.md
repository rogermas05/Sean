# Large PDF RAG System

A Retrieval-Augmented Generation (RAG) system optimized for processing and querying extremely large PDF documents (1000+ pages).

## Features

- Memory-efficient PDF processing in batches
- Page-by-page extraction to handle very large documents
- Chunk-based text splitting with configurable parameters
- Persistent vector database for fast querying
- Interactive query interface

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Install required packages:
   ```bash
   pip3 install -r requirements.txt
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Step 1: Process a Large PDF

Process your large PDF document and create a vector database:

```bash
python large_pdf_processor.py --pdf finalCopy.pdf --db ./your_document_db
```

Options:
- `--pdf`: Path to the PDF file (required)
- `--db`: Directory to store the vector database (default: ./pdf_vectordb)
- `--batch-size`: Number of pages to process in each batch (default: 100)
- `--chunk-size`: Size of text chunks (default: 1000)
- `--chunk-overlap`: Overlap between chunks (default: 200)
- `--force`: Force rebuild of vector store if it already exists

For very large documents (4000+ pages), you might want to reduce the batch size:

```bash
python large_pdf_processor.py --pdf your_large_document.pdf --db ./your_document_db --batch-size 50
```

### Step 2: Query the PDF

Once your document is processed, you can ask questions about it:

```bash
python rag_query.py --db ./your_document_db --question "What is the main topic of this document?"
```

For interactive mode (multiple questions):

```bash
python rag_query.py --db ./your_document_db
```

Options:
- `--db`: Path to the vector store directory (required)
- `--question`: Question to ask (runs in interactive mode if not provided)
- `--model`: OpenAI model to use (default: gpt-4o)
- `--temperature`: Temperature for response generation (default: 0)
- `--k`: Number of document chunks to retrieve (default: 5)
- `--no-sources`: Don't show source documents

#!/usr/bin/env python3
"""
RAG Query System - Question Answering for Large PDFs

This script queries a vector database created by large_pdf_processor.py
to answer questions about large PDF documents using RAG (Retrieval-Augmented Generation).
"""

import os
import argparse
from typing import Dict, Any

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def load_vectorstore(db_path: str) -> Chroma:
    """Load a persisted vector store."""
    print(f"Loading vector store from {db_path}...")
    
    # Check if directory exists
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please run large_pdf_processor.py first.")
    
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=db_path, 
        embedding_function=embeddings
    )

def setup_qa_chain(
    vectorstore: Chroma, 
    model_name: str = "gpt-4o", 
    temperature: float = 0,
    k: int = 5
) -> RetrievalQA:
    """Set up the question answering chain."""
    print(f"Setting up QA chain with model {model_name}...")
    
    template = """You are an assistant specialized in finding information in large documents.
    
    Use the following pieces of context to answer the question. The context comes from a large tariff/trade document.
    Try to provide as much relevant information as possible based on the context provided. If the information isn't complete,
    synthesize what you can from the available context.
    
    If you truly cannot find ANY relevant information in the context, only then say "I don't have enough information to answer this question."
    
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

def answer_question(qa_chain: RetrievalQA, question: str) -> Dict[str, Any]:
    """Answer a question using the QA chain."""
    return qa_chain.invoke({"query": question})

def display_answer(result: Dict[str, Any], show_sources: bool = True) -> None:
    """Display the answer and sources nicely."""
    print("\n" + "="*80)
    print("ANSWER:")
    print("-"*80)
    print(result["result"])
    
    if show_sources:
        print("\n" + "="*80)
        print("SOURCES:")
        print("-"*80)
        for i, doc in enumerate(result["source_documents"]):
            print(f"Source {i+1}:")
            print(f"{doc.page_content[:300]}..." if len(doc.page_content) > 300 else doc.page_content)
            print("-"*40)

def main():
    parser = argparse.ArgumentParser(
        description="RAG Query System for Large PDF Documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--db", required=True, help="Path to the vector store directory")
    parser.add_argument("--question", help="Question to ask (interactive mode if not provided)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for response generation")
    parser.add_argument("--k", type=int, default=5, help="Number of document chunks to retrieve")
    parser.add_argument("--no-sources", action="store_true", help="Don't show source documents")
    args = parser.parse_args()
    
    # Check if OPENAI_API_KEY is set
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it with: export OPENAI_API_KEY=your_api_key")
        return
    
    try:
        vectorstore = load_vectorstore(args.db)
        qa_chain = setup_qa_chain(
            vectorstore, 
            model_name=args.model, 
            temperature=args.temperature,
            k=args.k
        )
        
        if args.question:
            # Single question mode
            print(f"\nQuestion: {args.question}")
            result = answer_question(qa_chain, args.question)
            display_answer(result, show_sources=not args.no_sources)
        else:
            # Interactive mode
            print("\nEntering interactive mode. Type 'exit' to quit.")
            while True:
                question = input("\nEnter your question: ")
                if question.lower() in ("exit", "quit", "q"):
                    break
                
                result = answer_question(qa_chain, question)
                display_answer(result, show_sources=not args.no_sources)
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
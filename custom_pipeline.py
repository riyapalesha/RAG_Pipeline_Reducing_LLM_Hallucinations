import os
import tempfile
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community import embeddings
from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.doc import partition_doc
from unstructured.partition.text import partition_text
from unstructured.cleaners.core import group_broken_paragraphs
from unstructured.partition.docx import partition_docx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from io import BytesIO
from typing import List, Set, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer, util
import torch
import json
import sys
from dotenv import load_dotenv
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

# Constants
PERSISTENCE_DIR = Path("./demostore")
VECTORSTORE_PATH = PERSISTENCE_DIR / "faiss_index"
PROCESSED_FILES_PATH = PERSISTENCE_DIR / "processed_files.json"
FILE_PATH = "monopoly.pdf"  # Default file path

# Initialize models only once at the module level
SENTENCE_TRANSFORMER = SentenceTransformer('all-MiniLM-L6-v2')
QUERY_CACHE: Dict[str, Tuple[str, int, str, List[Document]]] = {}

def process_file(file_path):
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == 'pdf':
        elements = partition_pdf(
            filename=file_path,
            strategy="ocr_only",
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,  # Reduced from 8000 for better chunking
            combine_text_under_n_chars=200
        )
        texts = [str(element) for element in elements if hasattr(element, 'text')]
        return [Document(page_content=text, metadata={"source": file_path}) for text in texts]

    elif file_extension == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        elements = partition_text(
            text=text_content,
            paragraph_grouper=group_broken_paragraphs,
            max_partition=2000  # Reduced from 2500
        )
        texts = [str(element) for element in elements if hasattr(element, 'text')]
        return [Document(page_content=text, metadata={"source": file_path}) for text in texts]

    elif file_extension in ['docx', 'doc']:
        partition_func = partition_doc if file_extension == 'doc' else partition_docx
        elements = partition_func(
            filename=file_path,
            chunking_strategy="by_title",
            max_characters=4000,  # Reduced from 8000
            strategy="auto",
            infer_table_structure=True,
            combine_text_under_n_chars=800,
            chunk_overlap=300  # Reduced from 500
        )
        texts = [str(element) for element in elements if hasattr(element, 'text')]
        return [Document(page_content=text, metadata={"source": file_path}) for text in texts]

    elif file_extension == 'csv':
        df = pd.read_csv(file_path)
        documents = []
        for _, row in df.iterrows():
            content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            doc = Document(
                page_content=content,
                metadata={"source": file_path, "row": _}
            )
            documents.append(doc)
        return documents
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def semantic_rerank(query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
    # Using the global model instance for better performance
    query_embedding = SENTENCE_TRANSFORMER.encode(query, convert_to_tensor=True)

    # Batch processing for better performance
    doc_embeddings = SENTENCE_TRANSFORMER.encode(
        [doc.page_content for doc in documents],
        convert_to_tensor=True,
        batch_size=8,  # Add batch size for efficiency
        show_progress_bar=False  # Disable progress bar
    )

    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(documents)))

    return [documents[i] for i in top_results.indices]

def calculate_confidence_score(llm_answer, context_chunks):
    # Skip confidence scoring if SKIP_CONFIDENCE is True
    if os.environ.get("SKIP_CONFIDENCE", "false").lower() == "true":
        return 80, "Confidence scoring skipped to improve performance."

    fact_check_llm = Ollama(model="llama3.1:8b-instruct-q4_0", num_gpu=1, num_thread=4)

    context = "\n\n".join([chunk.page_content for chunk in context_chunks[:3]])  # Limiting to top 3 chunks

    fact_check_template = """
    Based only on the context, evaluate the accuracy of this answer:

    Context: {context}
    Answer: {answer}

    Score (0-100): [NUMBER]
    Explanation: [BRIEF]
    """

    fact_check_prompt = ChatPromptTemplate.from_template(fact_check_template)

    fact_check_chain = (
        fact_check_prompt
        | fact_check_llm
        | StrOutputParser()
    )

    fact_check_result = fact_check_chain.invoke({
        "context": context,
        "answer": llm_answer
    })

    try:
        score_line = [line for line in fact_check_result.split('\n') if 'Score' in line][0]
        score = int(score_line.split(':')[1].strip().split()[0])
        explanation = fact_check_result.split('Explanation:')[1].strip() if 'Explanation:' in fact_check_result else ""
        return score, explanation
    except (IndexError, ValueError):
        return 75, "Unable to parse confidence score. Using default."

def save_processed_files(processed_files: Set[str]):
    PERSISTENCE_DIR.mkdir(exist_ok=True)
    with open(PROCESSED_FILES_PATH, "w") as f:
        json.dump(list(processed_files), f)

def load_processed_files() -> Set[str]:
    if PROCESSED_FILES_PATH.exists():
        with open(PROCESSED_FILES_PATH, "r") as f:
            return set(json.load(f))
    return set()

def save_vectorstore(vectorstore: FAISS):
    PERSISTENCE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

def load_vectorstore() -> FAISS:
    if VECTORSTORE_PATH.exists():
        embeddings_model = embeddings.OllamaEmbeddings(model="nomic-embed-text")
        try:
            return FAISS.load_local(VECTORSTORE_PATH, embeddings_model, allow_dangerous_deserialization=True)
        except ValueError as e:
            print(f"Error loading vectorstore: {str(e)}")
            print("Creating new vectorstore.")
            return None
    return None

def process_and_update_vectorstore(file_path, processed_files: Set[str], vectorstore: FAISS = None) -> FAISS:
    embeddings_model = embeddings.OllamaEmbeddings(model="nomic-embed-text")
    new_documents = []

    file_name = os.path.basename(file_path)
    if file_name not in processed_files:
        print(f"Processing new file: {file_name}")
        processed_docs = process_file(file_path)
        new_documents.extend(processed_docs)
        processed_files.add(file_name)
        print(f"Added {len(processed_docs)} chunks from {file_name}")

    if vectorstore is None:
        if new_documents:
            vectorstore = FAISS.from_documents(documents=new_documents, embedding=embeddings_model)
        else:
            vectorstore = FAISS.from_documents(documents=[Document(page_content="Placeholder document")], embedding=embeddings_model)
    elif new_documents:
        vectorstore.add_documents(new_documents)

    if new_documents:
        save_vectorstore(vectorstore)
        save_processed_files(processed_files)

    return vectorstore

def ask_question(question, vectorstore):
    # Check cache first
    if question in QUERY_CACHE:
        print("Using cached response")
        return QUERY_CACHE[question]

    llm = Ollama(model="llama3.1:8b-instruct-q4_0", num_gpu=1, num_thread=4)

    after_rag_template = """Answer the question based only on the context below:
    {context}

    Question: {question}

    If the answer isn't in the context, say you don't know.
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # -------- UPDATED FUNCTION --------
    def retrieve_and_rerank(query):
        # FAISS search timing
        vs_start_time = time.time()
        docs = vectorstore.similarity_search(query, k=15)
        vs_time = time.time() - vs_start_time

        if not docs:
            print("No relevant documents found!")
            return "", [], vs_time, 0

        # Deduplicate
        seen = set()
        unique_docs = []
        for doc in docs:
            doc_identifier = hash(doc.page_content[:100])
            if doc_identifier not in seen:
                unique_docs.append(doc)
                seen.add(doc_identifier)

        # Rerank timing
        rerank_start_time = time.time()
        reranked_docs = semantic_rerank(query, unique_docs, top_k=5)
        rerank_time = time.time() - rerank_start_time

        print(f"[Timing] FAISS: {vs_time:.4f}s | Rerank: {rerank_time:.4f}s | Docs: {len(docs)}")

        return format_docs(reranked_docs), reranked_docs, vs_time, rerank_time

    # -------- RUN RETRIEVAL --------
    context_str, reranked_docs, vs_time, rerank_time = retrieve_and_rerank(question)

    if not context_str:
        return "I couldn't find relevant information to answer your question.", 0, "No context available", []

    rag_chain = (
        after_rag_prompt
        | llm
        | StrOutputParser()
    )

    # -------- LLM TIMING --------
    llm_start_time = time.time()
    llm_response = rag_chain.invoke({
        "context": context_str,
        "question": question
    })
    llm_time = time.time() - llm_start_time

    print(f"[Timing] LLM: {llm_time:.4f}s")

    # -------- CONFIDENCE --------
    score, explanation = calculate_confidence_score(llm_response, reranked_docs)

    # -------- CACHE --------
    QUERY_CACHE[question] = (llm_response, score, explanation, reranked_docs)

    return llm_response, score, explanation, reranked_docs

def optimize_ollama():
    """Ensure Ollama is configured optimally"""
    # These would typically be run in your notebook before running the script
    print("Note: For best performance in Colab, run these commands before script:")
    print("!nvidia-smi  # Check GPU availability")
    print("!ollama serve &  # Start Ollama server in background")
    print("You may also want to pull optimized models:")
    print("!ollama pull llama3.1:8b-instruct-q4_0")
    print("!ollama pull nomic-embed-text")

def main():
    # Show optimization tips
    optimize_ollama()

    # Check if file path is provided as a command-line argument
    file_path = sys.argv[1] if len(sys.argv) > 1 else FILE_PATH
    print(f"Processing file: {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Load or create vector store
    processed_files = load_processed_files()
    vectorstore = load_vectorstore()

    # Process the file and update the vectorstore
    vectorstore = process_and_update_vectorstore(file_path, processed_files, vectorstore)
    print("File processing complete. Vector store ready.")

    # Enable/disable confidence scoring
    confidence_option = input("Enable confidence scoring? (slower) [y/N]: ").lower()
    os.environ["SKIP_CONFIDENCE"] = "false" if confidence_option == 'y' else "true"

    # Interactive query loop
    while True:
        question = input("\nEnter your question (or 'exit' to quit): ")
        if question.lower() in ['exit', 'quit', 'q']:
            break

        print("\nProcessing...")
        start_time = time.time()
        llm_response, score, explanation, top_chunks = ask_question(question, vectorstore)
        total_time = time.time() - start_time

        print("\n----- ANSWER -----")
        print(llm_response)

        if os.environ.get("SKIP_CONFIDENCE", "true").lower() != "true":
            print("\n----- FACT CHECK -----")
            print(f"Confidence Score: {score}/100")
            print(f"Explanation: {explanation}")

        print(f"\n[Total Time] {total_time:.2f} seconds")

        # Ask if user wants to see the source chunks
        show_sources = input("\nShow source chunks? (y/n): ")
        if show_sources.lower() == 'y':
            print("\n----- TOP SOURCES -----")
            for i, doc in enumerate(top_chunks[:3], 1):  # Limit to top 3 for brevity
                print(f"\n{i}. {'-' * 40}")
                print(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))

if __name__ == "__main__":
    main()
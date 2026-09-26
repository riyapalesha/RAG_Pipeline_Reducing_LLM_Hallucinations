# Reducing Hallucinations in Large Language Models Using Retrieval-Augmented Generation

## Overview

Large Language Models (LLMs) are capable of generating fluent and informative responses, but they can also produce **hallucinations** responses that are factually incorrect, unsupported, or inconsistent with available information.

This project investigates a **Retrieval-Augmented Generation (RAG)** approach for reducing hallucinations by grounding LLM responses in information retrieved from user-provided documents.

The system combines:

* Document preprocessing and intelligent chunking
* Vector-based information retrieval
* Semantic re-ranking
* Context-grounded LLM generation
* Automated confidence scoring and fact checking
* Response-time measurement
* Query-response caching
* Persistent vector-store management

The project also evaluates the use of **FAISS and ChromaDB** as vector-store backends to study their effect on retrieval performance and the overall reliability of generated responses.


## Research Objective

The primary objective of this project is to investigate whether grounding LLM responses in externally retrieved information can reduce hallucinations and improve factual reliability.

The project specifically explores:

1. How effectively RAG grounds LLM responses in source documents.
2. Whether semantic re-ranking improves the relevance of retrieved context.
3. Whether automated confidence scoring can provide an additional indication of answer reliability.
4. The performance characteristics of different vector-store implementations, particularly **FAISS and ChromaDB**.
5. The relationship between retrieval performance, response latency, and factual reliability.


## System Architecture

The implemented pipeline follows the general workflow:

```text
                    ┌──────────────────────┐
                    │   Input Documents    │
                    │ PDF / DOCX / TXT /   │
                    │ CSV                  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Document Processing  │
                    │ & Chunking            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Embedding Generation  │
                    │ nomic-embed-text      │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │       Vector Store              │
              │                                 │
              │       FAISS / ChromaDB          │
              └────────────────┬────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     User Query       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Similarity Retrieval │
                    │      Top-15          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Semantic Re-ranking  │
                    │      Top-5           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Context-Grounded LLM │
                    │   Llama 3.1 8B       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Confidence Scoring & │
                    │    Fact Checking     │
                    └──────────────────────┘
```

## Methodology

### 1. Document Processing

The system accepts multiple document formats:

* PDF
* DOCX
* DOC
* TXT
* CSV

The `Unstructured` library is used to extract and partition content from documents. For PDF, DOCX, and DOC files, content is divided into chunks using title-based chunking while preserving relevant document structure.

For example, the current implementation uses a maximum chunk size of 4000 characters for PDF and DOC/DOCX documents, with overlapping chunks used for DOC/DOCX processing.

CSV files are processed row-by-row, with each row converted into a document containing its column-value pairs.


### 2. Embedding Generation

Document chunks are converted into vector representations using the **`nomic-embed-text`** embedding model through Ollama.

These vector representations allow documents to be compared according to their semantic meaning rather than relying only on exact keyword matches.


### 3. Vector Storage and Retrieval

The primary implementation uses **FAISS (Facebook AI Similarity Search)** to store and retrieve document embeddings.

The vector store is persisted locally, allowing it to be reused between executions rather than rebuilding the index for every run. The system also maintains a JSON file containing the names of previously processed files.

A corresponding **ChromaDB implementation** was developed as part of the research to compare vector-store alternatives under the same RAG framework.


### 4. Similarity Retrieval

For each user query, the vector store retrieves the **15 most relevant document chunks**.

The retrieved documents are then deduplicated before being passed to the semantic re-ranking stage.


### 5. Semantic Re-ranking

The initially retrieved documents are further ranked using the **`all-MiniLM-L6-v2` Sentence Transformer model**.

The query and retrieved documents are converted into embeddings, and cosine similarity is calculated between the query embedding and document embeddings. The five highest-scoring documents are selected as the final context.

```text
Initial Retrieval
       │
       ▼
   Top 15 chunks
       │
       ▼
 Deduplication
       │
       ▼
Semantic Similarity
       │
       ▼
    Top 5 chunks
```

The implementation performs the embedding calculations in batches and uses cosine similarity to select the highest-ranking documents.


### 6. Context-Grounded Response Generation

The selected document chunks are provided to **Llama 3.1 8B Instruct**, running locally through Ollama.

The generation prompt explicitly instructs the model to answer using only the retrieved context and to state that it does not know the answer when the required information is absent from the context.

This grounding mechanism is intended to reduce the model's reliance on unsupported parametric knowledge.


### 7. Confidence Scoring and Fact Checking

An optional confidence-scoring stage evaluates the generated response against the retrieved context.

A second Llama 3.1 8B instance is provided with the generated answer and the top three retrieved chunks. It is instructed to assign a score from **0–100** and provide a brief explanation of its assessment.

The confidence-scoring stage can be disabled when faster response times are preferred.

> **Note:** The confidence score is an LLM-generated assessment of contextual consistency; it should not be interpreted as a calibrated probability of correctness.


### 8. Performance Measurement

The system measures the time required for different stages of the pipeline, including:

* Vector-store retrieval time
* Semantic re-ranking time
* LLM generation time
* Total response time

These measurements allow retrieval and generation performance to be compared across different configurations.


### 9. Response Caching

Previously processed queries are stored in an in-memory cache.

If the same query is submitted again during execution, the previously generated response and associated information can be returned without repeating the complete retrieval and generation pipeline.


## Technologies Used

| Component               | Technology            |
| ----------------------- | --------------------- |
| Programming Language    | Python                |
| LLM Framework           | LangChain             |
| Local LLM Runtime       | Ollama                |
| Generative Model        | Llama 3.1 8B Instruct |
| Embedding Model         | nomic-embed-text      |
| Semantic Re-ranking     | Sentence Transformers |
| Re-ranking Model        | all-MiniLM-L6-v2      |
| Vector Store            | FAISS                 |
| Comparison Vector Store | ChromaDB              |
| Document Processing     | Unstructured          |
| Data Processing         | Pandas                |
| Similarity Calculation  | Cosine Similarity     |
| Tensor Computation      | PyTorch               |



## The default document configured in the attached implementation is `monopoly.pdf`, although a different document can be supplied through the command line.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Python dependencies

Install the required packages using the project's dependency file if one is provided:

```bash
pip install -r requirements.txt
```

If a requirements file is not included, the major dependencies used by the implementation include:

```bash
pip install langchain langchain-community
pip install sentence-transformers
pip install torch
pip install pandas
pip install unstructured
pip install python-dotenv
```

Additional dependencies may be required by `Unstructured` depending on the document formats being processed.


## Ollama Setup

This project uses Ollama for local LLM inference and embeddings.

Start the Ollama service:

```bash
ollama serve
```

Pull the required models:

```bash
ollama pull llama3.1:8b-instruct-q4_0
ollama pull nomic-embed-text
```

The implementation is configured to use the quantized `llama3.1:8b-instruct-q4_0` model with GPU and CPU-thread settings.



## Running the Project

Place the document you want to process in the project directory.

Run the pipeline using:

```bash
python custom_pipeline.py
```

Alternatively, provide a document path:

```bash
python custom_pipeline.py <path-to-document>
```

The program processes the document, creates or loads the vector store, and then starts an interactive question-answering loop.


## Example Workflow

```text
1. Provide a document
        ↓
2. Extract and chunk document content
        ↓
3. Generate embeddings
        ↓
4. Store embeddings in vector database
        ↓
5. Enter a question
        ↓
6. Retrieve top 15 relevant chunks
        ↓
7. Remove duplicate chunks
        ↓
8. Re-rank using semantic similarity
        ↓
9. Select top 5 chunks
        ↓
10. Generate grounded answer
        ↓
11. Optionally perform confidence scoring
        ↓
12. Display answer, timing, and confidence information
```



## Example Output

A typical interaction follows this structure:

```text
Enter your question:

Processing...

[Timing] FAISS: 0.00XXs | Rerank: 0.0XXXs | Docs: 15
[Timing] LLM: X.XXXXs

----- ANSWER -----

<LLM-generated answer based on retrieved context>

----- FACT CHECK -----

Confidence Score: XX/100
Explanation: <brief explanation>

[Total Time] XX.XX seconds
```

The system can also display the top retrieved source chunks, allowing users to inspect the evidence used to generate the answer.



## FAISS vs. ChromaDB

A major component of the research is the comparison between **FAISS and ChromaDB** as vector-store backends for the RAG pipeline.

The comparison is performed while keeping the major components of the pipeline consistent, allowing the vector-store implementation to be studied independently.

The evaluation considers factors such as:

* Retrieval latency
* Retrieval relevance
* Response latency
* Factual consistency
* Hallucination frequency
* Confidence scores
* Storage and scalability characteristics

The results of these experiments are reported in the research paper accompanying this project.



## Key Features

* **Multi-format document support** — PDF, DOC, DOCX, TXT, and CSV.
* **Retrieval-Augmented Generation** — grounds responses in retrieved documents.
* **Semantic re-ranking** — improves the relevance of retrieved context.
* **Local LLM inference** — uses Ollama rather than requiring a remote API.
* **Confidence assessment** — optionally evaluates generated answers against retrieved context.
* **Persistent vector storage** — avoids rebuilding the vector store unnecessarily.
* **Incremental document processing** — tracks previously processed files.
* **Performance monitoring** — records retrieval, re-ranking, LLM, and total response times.
* **Source inspection** — allows users to view the top retrieved chunks.
* **FAISS and ChromaDB comparison** — evaluates alternative vector-store implementations.



## Limitations

The current implementation has several limitations:

1. The confidence score is generated by an LLM and is therefore not equivalent to a statistically calibrated probability of correctness.
2. The quality of generated responses depends on the quality and completeness of the source documents.
3. Retrieval errors can propagate to the generation stage.
4. Semantic re-ranking introduces additional computational overhead.
5. The evaluation is dependent on the selected document collection and question set.
6. The current implementation uses a relatively small local LLM rather than evaluating multiple model families.
7. Performance measurements may vary depending on hardware and GPU availability.



## Research Contribution

This project explores a practical approach to hallucination mitigation by combining **retrieval augmentation, semantic re-ranking, context-grounded generation, and confidence assessment** within a single pipeline.

In addition to studying hallucination reduction, the project investigates how the choice of vector-store backend can influence the performance and reliability of a RAG-based LLM system.



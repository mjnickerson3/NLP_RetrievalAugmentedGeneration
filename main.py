"""
Retrieval-Augmented Question Answering System
Student Starter Template

Requirements:
pip install scikit-learn

Files needed:
corpus.txt

Instructions:
- Complete all TODO sections
- Keep each paragraph in corpus.txt separated by a blank line
- Test at least 3 different queries
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_FILE = "corpus.txt"
TOP_K = 3


@dataclass
class RetrievalResult:
    document_index: int
    score: float
    text: str


# --------------------------------------------------
# PART 1: LOAD CORPUS
# --------------------------------------------------
def load_corpus(filepath: str) -> List[str]:
    """
    Load the corpus and split it into paragraph-level documents.
    """
    path = Path(filepath)

    # Check whether the file exists
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {filepath}. Please create the file in the project folder."
        )

    # Read the file contents
    raw_text = path.read_text(encoding="utf-8")

    # Make sure the corpus is not empty
    if not raw_text.strip():
        raise ValueError("The corpus file is empty.")

    # Split the corpus into documents using blank lines
    documents = [doc.strip() for doc in raw_text.split("\n\n") if doc.strip()]

    # Return the list of documents
    return documents


# --------------------------------------------------
# PART 2: BUILD RETRIEVER
# --------------------------------------------------
def build_retriever(documents: List[str]) -> Tuple[TfidfVectorizer, object]:
    """
    Create a TF-IDF vectorizer and fit it on the document collection.
    """

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the documents
    document_matrix = vectorizer.fit_transform(documents)

    return vectorizer, document_matrix


# --------------------------------------------------
# PART 3: RETRIEVE DOCUMENTS
# --------------------------------------------------
def retrieve_documents(
    query: str,
    documents: List[str],
    vectorizer: TfidfVectorizer,
    document_matrix,
    top_k: int = 3,
) -> List[RetrievalResult]:
    """
    Rank documents for a query using cosine similarity over TF-IDF vectors.
    """

    # Convert the query into a TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Compute cosine similarities between query and document matrix
    similarities = cosine_similarity(
        query_vector,
        document_matrix
    ).flatten()

    # Get the indices of the top_k highest scores
    ranked_indices = similarities.argsort()[-top_k:][::-1]

    results: List[RetrievalResult] = []

    # Build a list of RetrievalResult objects
    for idx in ranked_indices:
        results.append(
            RetrievalResult(
                document_index=int(idx),
                score=float(similarities[idx]),
                text=documents[idx],
            )
        )

    return results


# --------------------------------------------------
# PART 4: ANSWER WITHOUT RETRIEVAL
# --------------------------------------------------
def answer_without_retrieval(query: str) -> str:
    """
    Simulated answer without retrieval.
    In a future upgrade, this could call an LLM directly.
    """

    return (
        "This answer was generated without retrieved documents.\n\n"
        f"Question: {query}\n"
        "Answer: This response is based only on general knowledge and may be incomplete or inaccurate."
    )


# --------------------------------------------------
# PART 5: BUILD CONTEXT
# --------------------------------------------------
def build_context(results: List[RetrievalResult]) -> str:
    """
    Combine retrieved passages into one context block.
    """

    context_parts = []

    # Loop through the results and build a context string
    for rank, result in enumerate(results, start=1):
        context_parts.append(
            f"[Document {rank} | Paragraph #{result.document_index + 1} | Score: {result.score:.4f}]\n"
            f"{result.text}"
        )

    return "\n\n".join(context_parts)


# --------------------------------------------------
# PART 6: ANSWER WITH RETRIEVAL
# --------------------------------------------------
def answer_with_retrieval(query: str, results: List[RetrievalResult]) -> str:
    """
    Simple grounded answer generator.
    This version uses retrieved context and a template-based response.
    """

    if not results:
        return "No relevant documents were retrieved."

    context = build_context(results)

    # Replace this with your own grounded answer template
    answer = (
        "Grounded Answer\n"
        "--------------\n"
        f"Question: {query}\n\n"
        "Retrieved Context:\n"
        f"{context}\n\n"
        "Response:\n"
        "The answer above is based on the retrieved QA and testing documents. "
        "Using retrieved information provides more relevant and grounded answers "
        "than relying only on general knowledge."
    )

    return answer


# --------------------------------------------------
# PART 7: DISPLAY HELPERS
# --------------------------------------------------
def print_retrieval_results(results: List[RetrievalResult]) -> None:
    """
    Print ranked retrieval results in a readable way.
    """

    print("\nTop Retrieved Documents:")
    print("-----------------------")

    if not results:
        print("No documents found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Corpus Paragraph #: {result.document_index + 1}")
        print(f"Similarity Score : {result.score:.4f}")
        print(f"Text : {result.text}")


def compare_outputs(no_retrieval_answer: str, retrieval_answer: str) -> None:
    """
    Print both answers so students can compare them.
    """

    print("\nComparison:")
    print("-----------")

    print("Without Retrieval:")
    print(no_retrieval_answer)

    print("\nWith Retrieval:")
    print(retrieval_answer)

    print("\nTODO: In your README, explain which version is better and why.")


# --------------------------------------------------
# PART 8: RUN ONE QUERY
# --------------------------------------------------
def run_query(
    query: str,
    documents: List[str],
    vectorizer: TfidfVectorizer,
    document_matrix,
    top_k: int = TOP_K,
) -> None:
    """
    Run the full pipeline for one query.
    """

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    #  Get answer without retrieval
    no_retrieval_answer = answer_without_retrieval(query)

    print("\n1) Answer WITHOUT Retrieval")
    print("---------------------------")
    print(no_retrieval_answer)

    #  Retrieve top documents
    results = retrieve_documents(
        query=query,
        documents=documents,
        vectorizer=vectorizer,
        document_matrix=document_matrix,
        top_k=top_k,
    )

    print_retrieval_results(results)

    # Get answer with retrieval
    retrieval_answer = answer_with_retrieval(query, results)

    print("\n2) Answer WITH Retrieval")
    print("------------------------")
    print(retrieval_answer)

    compare_outputs(no_retrieval_answer, retrieval_answer)


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------
def main() -> None:

    # Load corpus
    documents = load_corpus(CORPUS_FILE)

    print(f"Loaded {len(documents)} documents from {CORPUS_FILE}.")

    #  Build TF-IDF retriever
    vectorizer, document_matrix = build_retriever(documents)

    # Replace with your own test queries
    queries = [
        "What is Playwright?",
        "How does test automation help QA teams?",
        "Why is regression testing important?",
    ]

    for query in queries:
        run_query(
            query,
            documents,
            vectorizer,
            document_matrix,
            top_k=TOP_K
        )

    print("\nAssignment Reminder:")
    print("- Test at least 3 queries")
    print("- Compare answers with and without retrieval")
    print("- Include screenshots in your document")
    print("- Explain when retrieval helped and when it failed")


if __name__ == "__main__":
    main()
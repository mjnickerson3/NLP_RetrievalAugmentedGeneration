# NLP_RetrievalAugmentedGeneration

## Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) system using TF-IDF and cosine similarity. The system retrieves the most relevant documents from a custom corpus focused on quality assurance, test automation, Playwright, AI agents, and software testing workflows before generating a response. The goal is to show how retrieval can improve answer quality by providing supporting context instead of relying only on general knowledge.

## Files Included

- main.py
- corpus.txt
- README.md

## Example Queries

1. What is Playwright?
2. How does test automation help QA teams?
3. Why is regression testing important?

<img width="975" height="407" alt="image" src="https://github.com/user-attachments/assets/4bb3efd4-b84e-4c3a-8733-10815a9383ec" />

<img width="975" height="410" alt="image" src="https://github.com/user-attachments/assets/9049ebe2-b133-491f-80d9-a102682c65ea" />

<img width="975" height="412" alt="image" src="https://github.com/user-attachments/assets/cc4f28f6-dc60-4754-a6d7-c9cdfa03310d" />


## Analysis

The retrieval-based version performed better because it used information that was actually stored in the corpus instead of giving a generic response. The version without retrieval provided a very general answer, while the retrieval-based version returned relevant paragraphs and used them to support the response.

Retrieval helped the most when the question matched topics that were included in the corpus. For example when I asked about Playwright, the system returned the paragraph that described what Playwright is and how it is used for test automation. The questions about test automation and regression testing also returned relevant information that helped answer the question more accurately.

One limitation is that retrieval depends on the information available in the corpus. If someone asked a question about a topic that was not included, the results would likely be less useful. Since TF-IDF looks for matching words, different wording could also affect which documents get returned.

Most of the answers were accurate because they came directly from the corpus. However, some of the retrieved paragraphs were only partially related to the question. For example, when I asked about Playwright, one of the returned documents discussed AI agents because it also mentioned Playwright.

If I continued improving this project, I would add more documents to the corpus and look at ways to improve document ranking. I would also consider using a language model to generate more detailed answers from the retrieved information.

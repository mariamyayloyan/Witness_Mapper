"""
This script processes a list of contradiction entries and evaluates the most relevant chunks of 
text from associated legal documents to identify the likely cause of each contradiction.

It uses Azure OpenAI's GPT-4o model to:
- Score relevance of text chunks to the contradiction.
- Complete incomplete sentences if the most relevant chunk is cut off at the beginning or end.

The main function `evaluate_contradiction_sources` adds the most relevant chunk and a relevance 
score to each contradiction entry.

Environment Variables:
- AZURE_OPENAI_API_KEY: Azure API key for OpenAI.
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL.

Dependencies:
- os
- json
- re
- openai (Azure OpenAI via openai.AzureOpenAI)

"""

import json
import os
import re
from openai import AzureOpenAI


client_openai = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


def break_into_chunks(text):
    """
    Breaks a block of text into smaller, semantically meaningful chunks (paragraphs).

    It:
        - Merges lines that are likely mid-sentence (i.e., when a line starts with a lowercase letter or digit).
        - Keeps paragraph breaks when a line starts with a capital letter, bullet, or number.

    Args:
        text (str): Raw document text with newline characters.

    Returns:
        list of str: Cleaned and separated paragraphs from the original text.
    """
    text = re.sub(r'\n(?=[a-z0-9])', ' ', text)
    text = re.sub(r'\n(?=[A-Z•\(])', '\n\n', text)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    return paragraphs


def evaluate_contradiction_sources(contradictions, full_docs):
    """
    Identifies the most likely cause of contradictions by scoring chunks of associated document text
    and selecting the most relevant one for each contradiction.

    Skips contradictions that originate from tables (i.e., entries with a 'doc_type' field).

    For each eligible contradiction entry:
        - Breaks the document text into smaller chunks.
        - Uses a language model to score each chunk's relevance to the contradiction (scale: 1–10).
        - Identifies the most relevant chunk based on the highest score.
        - If the selected chunk ends mid-sentence, requests the model to complete it using the full document context,
          ensuring the chunk ends at a natural sentence boundary.

    Args:
        contradictions (list of dict): Each dictionary may include:
            - 'result' (str): The contradiction explanation text.
            - 'doc_text' (str): The full document text (not from tables).
            - 'doc_type' (str, optional): Present if the contradiction originates from a table.

    Returns:
        list of dict: Updated entries with two new keys:
            - 'most_relevant_chunk' (str or None): The chunk most likely causing the contradiction.
            - 'relevance_score' (int): The chunk’s relevance score (1–10). If skipped, set to 0.
    """
    results = []

    for i, entry in enumerate(contradictions):
        if "doc_type" in entry:
            print(
                f"Skipping entry {i} — contradiction originates from a table.")
            results.append(entry)
            continue

        contradiction_text = entry["result"]
        doc_text = entry["doc_text"]

        chunks = break_into_chunks(doc_text)
        best_score = -1
        best_chunk = ""

        for chunk in chunks:
            prompt = f"""Here is a contradiction:

{contradiction_text}

Evaluate the following chunk from a legal document to determine how likely it caused this contradiction.
Return only a number from 1 to 10 (1 = not relevant, 10 = directly caused it).

Chunk:
{chunk}"""

            try:
                response = client_openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a legal analyst evaluating legal contradictions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                score_text = response.choices[0].message.content.strip()
                score = int(re.search(r"\d+", score_text).group())

                if score > best_score:
                    best_score = score
                    best_chunk = chunk

            except Exception as e:
                print(f"Error at entry {i}: {e}")
                continue
        prev_page = ""
        next_page = ""
        for doc in full_docs:
            if doc.get("section") == entry.get("doc_section") and doc.get("page") == entry.get("doc_page") - 1:
                prev_page = doc.get("text", "")
            if doc.get("section") == entry.get("doc_section") and doc.get("page") == entry.get("doc_page") + 1:
                next_page = doc.get("text", "")
            if prev_page and next_page:
                break
        # Checking if the most relevant chunk seems like an incomplete sentence and completing using GPT-4o
        if best_chunk and (not best_chunk.endswith(('.', '!', '?')) or not best_chunk[0].isupper()):
            completion_prompt = f"""If the following chunk seems to have an unfinished sentence, or lacks context:

{best_chunk}

Please complete the sentence, making sure it ends at the nearest full stop and has a meaningful beginning.
Consider the text from the full document to complete it:

Document text:
{doc_text}

If the first sentence is in the beginning of the document page and got cut between pages, consider also the previous page:
{prev_page}

If the last sentence is in the end of the document page and got cut between pages, consider also the next page:
{next_page}

Return only the completed text chunk."""

            try:
                completion_response = client_openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a legal analyst evaluating legal contradictions."},
                        {"role": "user", "content": completion_prompt}
                    ],
                    temperature=0.2
                )
                completed_chunk = completion_response.choices[0].message.content.strip(
                )
                best_chunk = completed_chunk

            except Exception as e:
                print(f"Error while completing chunk at entry {i}: {e}")
                continue

        entry["most_relevant_chunk"] = best_chunk
        entry["relevance_score"] = best_score
        results.append(entry)

    return results


# Finding most relevant chunk for the meaningful contradictions
contr_path = '../results/correct_contradictions.json'
full_docs_path = '../results/combined_doc_sections_with_tables.json'

with open(contr_path, 'r', encoding='utf-8') as f:
    contradictions = json.load(f)

with open(full_docs_path, 'r', encoding='utf-8') as f:
    full_docs = json.load(f)

results_contr_chunks = evaluate_contradiction_sources(
    contradictions, full_docs)

# Saving in a JSON
with open("../results/final_correct_contr.json", "w", encoding="utf-8") as f:
    json.dump(results_contr_chunks, f, ensure_ascii=False, indent=4)


# Finding most relevant chunk for all contradictions
contr_path_flagged = '../results/flagged_contradictions.json'

with open(contr_path_flagged, 'r', encoding='utf-8') as f:
    contradictions_flagged = json.load(f)

results_contr_chunks_flagged = evaluate_contradiction_sources(
    contradictions_flagged, full_docs)

# Saving in a JSON
with open("../results/final_flagged_contr.json", "w", encoding="utf-8") as f:
    json.dump(results_contr_chunks_flagged, f, ensure_ascii=False, indent=4)

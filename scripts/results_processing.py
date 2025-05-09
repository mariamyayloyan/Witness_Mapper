"""
This script processes contradiction entries from two JSON files, merges them while avoiding duplicates,
and evaluates each contradiction entry against arbitration documents. It utilizes GPT-based models to filter
out irrelevant contradictions and verify the validity of the contradictions in the context of the documents.

The main functions in this script are:

1. `merge_unique_contradictions`: Merges contradiction entries from two files, eliminating duplicates based on section, 
   page, and witness statement. It uses OpenAI's o3-mini model to decide if two entries describe the same contradiction.
   
2. `add_titles_to_entries`: Adds section titles to contradiction entries based on a dictionary of titles for document sections
   and witness sections.
   
3. `check_contradictions_with_context`: For each contradiction, this function divides the full documents into chunks, extracts
   the most relevant context related to the contradiction from these chunks using a language model, and evaluates the contradiction's
   validity with respect to the context. It adds a 'contr_correct' field indicating whether the contradiction is valid ('yes' or 'no').

4. `filter_correct_contr`: Filters the contradictions that have been validated ('yes' in `contr_correct` field).

The final results are saved into two JSON files:
- `flagged_contradictions.json` contains contradictions with a context evaluation.
- `correct_contradictions.json` contains only the contradictions deemed correct and useful for legal purposes.

Parameters and outputs:
- `merge_unique_contradictions`: 
    - Inputs: `file1_path` and `file2_path` (paths to two JSON files containing contradiction data).
    - Output: Merged list of unique contradiction entries.
  
- `add_titles_to_entries`: 
    - Inputs: `merged_data` (list of contradiction entries), `titles` (dictionary of section titles).
    - Output: Updated list of contradiction entries with added section titles.
  
- `check_contradictions_with_context`: 
    - Inputs: `contradictions` (list of contradiction entries), `full_docs_path` (path to the full documents).
    - Output: List of contradiction entries with context evaluation (`contr_correct` field).
  
- `filter_correct_contr`: 
    - Input: `entries` (list of contradiction entries with `contr_correct` field).
    - Output: List of entries where `contr_correct` is 'yes'.

Environment Variables:
- AZURE_OPENAI_API_KEY: Azure API key for OpenAI.
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL.

Dependencies:
- `os`: Standard Python library for interacting with the operating system (used for loading environment variables).
- `json`: Standard Python library for parsing and working with JSON data.
- `re`: Standard Python library for regular expressions (used for extracting titles from section identifiers).
- `openai`: The OpenAI Python client library for interacting with the Azure OpenAI API (used to access GPT models).

"""

import os
import json
import re
from openai import AzureOpenAI


# Setting up the Azure OpenAI client
client_openai = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


def merge_unique_contradictions(file1_path, file2_path):
    """
    Merges contradiction entries from two JSON files, filtering out duplicates based on section, page, and 
    witness statement. If entries match on these keys but have different contradiction descriptions, it uses
    a language model to decide if they are describing the same contradiction.

    Args:
        file1_path (str): Path to the first JSON file.
        file2_path (str): Path to the second JSON file.

    Returns:
        list: The merged list of unique contradiction entries.
    """
    def filter_contradictions(data):
        """Returns only entries where 'result' includes a contradiction."""
        return [entry for entry in data if 'result' in entry and 'Contradiction:' in entry['result']]

    with open(file1_path, 'r', encoding='utf-8') as f1:
        data1 = filter_contradictions(json.load(f1))

    with open(file2_path, 'r', encoding='utf-8') as f2:
        data2 = filter_contradictions(json.load(f2))

    for entry2 in data2:
        match_found = False
        for entry1 in data1:
            same_reference = (
                entry1.get("doc_section") == entry2.get("doc_section") and
                entry1.get("doc_page") == entry2.get("doc_page") and
                entry1.get("witness_statement") == entry2.get(
                    "witness_statement")
            )

            if same_reference:
                messages = [
                    {"role": "system", "content": "You are a legal expert helping identify duplicate contradictions."},
                    {"role": "user", "content": (
                        f"Here are two contradiction descriptions from the same section, page, and witness:\n\n"
                        f"Contradiction 1:\n{entry1['result']}\n\n"
                        f"Contradiction 2:\n{entry2['result']}\n\n"
                        "Are these describing the same contradiction? Reply only with 'yes' or 'no'."
                    )}
                ]

                response = client_openai.chat.completions.create(
                    model="o3-mini",
                    messages=messages,
                    temperature=0
                )

                answer = response.choices[0].message.content.strip().lower()
                if "yes" in answer:
                    match_found = True
                    break

        if not match_found:
            data1.append(entry2)

    return data1


file1_path = "../results/comp_results1.json"
file2_path = "../results/comp_results2.json"

merged_data = merge_unique_contradictions(
    file1_path,
    file2_path
)


def add_titles_to_entries(merged_data, titles):
    """
    Adds section titles to each entry in the merged data based on the provided titles dictionary.
    The function adds titles for both the document sections (doc_section) and witness sections (witness_section),
    extracting titles where available.

    Args:
        merged_data (list): List of contradiction entries to be updated.
        titles (dict): Dictionary mapping section identifiers to their respective titles.

    Returns:
        list: The merged data with titles added for doc_section and witness_section.
    """
    for entry in merged_data:
        # Getting the section from the entry
        section_d = entry.get("doc_section")
        section_w = entry.get("witness_section")

        # Addding general title from doc_section
        entry["doc_title"] = titles.get(section_d, "Unknown Section")

        # Extracting title from section_w using the part in parentheses
        if section_w:
            match = re.search(r'\(([^)]+)\)', section_w)
            if match:
                key_in_titles = match.group(1)  # e.g., "Claimant Exhibit C 5"
                entry["witness_title"] = titles.get(
                    key_in_titles, "Unknown Witness Section")
            else:
                entry["witness_title"] = "Unknown Witness Section"

    return merged_data


# Defining the titles dictionary
titles = {
    "Letter by Langweiler": "Letter by Langweiler",
    "Request for Arbitration": "Request for Arbitration",
    "Claimant Exhibit C 1": "REQUEST FOR QUOTATION (RFQ)",
    "Claimant Exhibit C 2": "PURCHASE AND SERVICE AGREEMENT",
    "Claimant Exhibit C 3": "TRANSITION NEWS",
    "Claimant Exhibit C 4": "Email Re:Update on supplier",
    "Claimant Exhibit C 5": "Witness Statement Poul Cavendish",
    "Claimant Exhibit C 6": "Re: Termination of the Purchase and Service Agreement of 17 July 2023",
    "Claimant Exhibit C 7": "Without-prejudice Offer",
    "Letters by FAI": "Letters by FAI",
    "Letter by Fasttrack": "Letter by Fasttrack",
    "Answer to the Request for Arbitration": "Answer to the Request for Arbitration",
    "Respondent Exhibit R 1": "Witness Statement of Johanna Ritter",
    "Respondent Exhibit R 2": "Email Re:  Local content",
    "Respondent Exhibit R 3": "Email Re:  Local content",
    "Respondent Exhibit R 4": "News from the Bar",
    "Letter by Langweiler Objecting to Admittance of Document": "Letter by Langweiler Objecting to Admittance of Document",
    "Claimant Exhibit C 8": "Witness Statement of August Wilhelm Deiman",
    "Letters by FAI (2)": "Letters by FAI",
    "Letter by FAI Concerning the Decisions Made by the Board": "Letter by FAI Concerning the Decisions Made by the Board",
    "Letter by FAI Confirming the Party-nominated Arbitrators": "Letter by FAI Confirming the Party-nominated Arbitrators",
    "Letters by FAI Concerning the Appointment of Presiding Arbitrator": "Letters by FAI Concerning the Appointment of Presiding Arbitrator",
    "Letters by Greenhouse": "Letters by Greenhouse",
    "Procedural Order No. 1": "Procedural Order No. 1",
    "Procedural Order No. 2": "Procedural Order No. 2"
}

# Using the function on merged data
updated_data = add_titles_to_entries(merged_data, titles)


def check_contradictions_with_context(contradictions, full_docs_path):
    """
    For each contradiction:
    - GPT-4.1 scans 3 chunks of documents and finds relevant info.
    - If no relevant info is found across all chunks, skip the contradiction.
    - Otherwise, ask o3-mini if the contradiction makes sense in context.
    - Add 'contr_correct': 'yes' or 'no' to each valid contradiction.
    """
    with open(full_docs_path, 'r', encoding='utf-8') as f:
        full_docs = json.load(f)

    # Split into 3 parts
    n = len(full_docs)
    chunk_size = n // 3
    doc_chunks = [
        full_docs[i * chunk_size: (i + 1) *
                  chunk_size] if i < 2 else full_docs[i * chunk_size:]
        for i in range(3)
    ]

    updated_contradictions = []

    for idx, entry in enumerate(contradictions):
        entry_text = json.dumps(entry, indent=4, ensure_ascii=False)
        relevant_context_parts = []

        for i, chunk in enumerate(doc_chunks):
            chunk_text = "\n\n".join(
                f"[{doc.get('section', '')} - p.{doc.get('page', '')}]\n{doc.get('text', '')}"
                for doc in chunk
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are a legal assistant. Your job is to extract only the most relevant passages from the documents based on a contradiction entry."
                },
                {
                    "role": "user",
                    "content": f"""
Here is a contradiction entry:
{entry_text}

Here is a chunk of arbitration documents:
{chunk_text}

Please extract the most relevant text segments related to this contradiction (not limited to the witness statement and the exhibit that caused it). If there's nothing relevant, just skip. Don't include explanation, only text segments."
"""
                }
            ]

            response = client_openai.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                temperature=0
            )

            relevant = response.choices[0].message.content.strip()
            print(relevant)
            relevant_context_parts.append(
                f"--- From Chunk {i+1} ---\n{relevant}")

        if not relevant_context_parts:
            print(
                f"Skipped {idx + 1}/{len(contradictions)} — no relevant context.")
            continue

        final_context = "\n\n".join(relevant_context_parts)

        decision_messages = [
            {
                "role": "system",
                "content": "You are a legal expert assessing contradictions in arbitration documents."
            },
            {
                "role": "user",
                "content": f"""
Here is the contradiction entry:
{entry_text}

Here is all relevant context from the documents:
{final_context}

Does this contradiction truly indicate a misrepresentation or inconsistency when considered in full context?
Is it a valid contradiction that can be useful for the lawyers to use for cross-examination or evaluate where the witness is lying?
Take into account everything, including sections, titles of entries and result, which is the explanation of the contradiction. Be very critical, pay close attention to the explanation, make sure it's not 2 unrelated things wrongly identified as contradiction.
Ignore harmless or explainable changes (e.g., a change in CEO, company name, or location), especially if the context explicitly clarifies the situation (e.g., by saying 'then-CEO', 'formerly known as', etc.).
Only mark 'yes' if the contradiction is substantial and would actually help a lawyer identify a misrepresentation or inconsistency worth pursuing.
Reply only with 'yes' or 'no'."
"""
            }
        ]

        decision = client_openai.chat.completions.create(
            model="o3-mini",
            messages=decision_messages,
            temperature=0
        )

        entry["contr_correct"] = decision.choices[0].message.content.strip().lower()
        updated_contradictions.append(entry)

        print(
            f"Processed {idx + 1}/{len(contradictions)}: contr_correct = {entry['contr_correct']}")

    return updated_contradictions


def filter_correct_contr(entries):
    """
    Returns only entries where contr_correct is 'yes'.
    """
    return [entry for entry in entries if "yes" in entry.get("contr_correct", "").strip().lower()]


# Checking the contradictions with context
checked = check_contradictions_with_context(
    merged_data,
    full_docs_path="../results/combined_doc_sections_with_tables.json"
)

# Saving the results
with open("../results/flagged_contradictions.json", "w", encoding="utf-8") as f:
    json.dump(checked, f, indent=4, ensure_ascii=False)

# Keeping meaningful contradictions
correct_contr = filter_correct_contr(checked)

# Saving the results
with open("../results/correct_contradictions.json", 'w', encoding='utf-8') as f:
    json.dump(correct_contr, f, indent=4, ensure_ascii=False)

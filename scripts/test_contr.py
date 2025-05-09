"""
This script compares witness statements against pages from arbitration documents to identify contradictions, 
inconsistencies, or conflicts. The comparison is performed using OpenAI's Azure API, where the text of each 
witness statement is checked against a section of an arbitration document.

The script performs the following tasks:
1. Loads and filters the relevant arbitration document pages and witness statements.
2. Uses OpenAI's API to check for contradictions between the witness statement and the arbitration document.
3. Stores the results of the comparison, including any identified contradictions, in a JSON file.

The results are saved in a file named `comp_results13.json`, which contains the following information for each comparison:
- The witness statement text and its page and section.
- The arbitration document text and its page and section.
- The result of the contradiction check (either "Contradiction" or "Neutral").
- Metadata about the type of witness statement and arbitration document chunks (if it's a table).

Environment Variables:
- AZURE_OPENAI_API_KEY: Azure API key for OpenAI.
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL.

Dependencies:
- openai (Azure OpenAI via `openai.AzureOpenAI`)
- os, json
"""

import json
import os
from openai import AzureOpenAI

# Load Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


def check_contradiction(sentence, arbitration_document):
    """
    Compares a sentence (typically from a witness statement) against a page from an arbitration document 
    to detect any contradictions or inconsistencies using Azure OpenAI's GPT-4o model.

    Parameters:
        sentence (str): The sentence from a witness statement to evaluate.
        arbitration_document (str): The full page of the arbitration document to compare against.

    Returns:
        str: A result message indicating whether there is a contradiction ("Contradiction:") or not ("Neutral:"),
             followed by a concise explanation.
    """

    prompt = f"""
You are an expert in arbitration law and practice. Your task is to compare the following sentence with a page from an arbitration document (e.g., an exhibit) and identify any contradictions, inconsistencies, or conflicts. A contradiction occurs when the sentence directly opposes the findings, legal reasoning, or rulings stated in the arbitration document.

Please follow these steps:
1. Read both the sentence and the full page carefully.
2. Identify the key legal claims, defenses, arguments, and rulings in both the sentence and the page from the arbitration document.
3. Compare these claims and arguments, and determine if the sentence contradicts any key facts, findings, or conclusions in the arbitration document.
4. If a contradiction exists, start your response with the word "Contradiction:" and explain why the sentence contradicts the arbitration document in a clear and concise manner (about 20 words). If there is no contradiction, start your response with "Neutral:" and explain why there's no contradiction.
5. Do not assume differences between earlier proposals and later outcomes happened because both parties agreed to a change unless the document states so.

Sentence:
\"{sentence}\"

Page from an Arbitration Document:
\"{arbitration_document}\"
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
                "content": "You are an expert in arbitration law and practice."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )

    return response.choices[0].message.content.strip()


def detect_witness_contradictions(document_pages, witness_statements, excluded_sections, output_path):
    """
    Detects contradictions between witness statements and document pages.

    This function compares filtered witness statements to selected pages from arbitration documents,
    excluding specified sections. It uses the `check_contradiction` function to identify contradictions,
    then saves the results in a JSON file.

    Parameters:
        document_pages: List of dictionaries representing pages of the arbitration document.
        witness_statements: List of dictionaries representing witness statements.
        excluded_sections: Set of section names to exclude from comparison.
        output_path: File path to save the output JSON containing contradiction results.

    Returns:
        None. Outputs a JSON file at the specified path.
    """

    # Filtering out excluded document sections
    filtered_docs = [entry for entry in document_pages
                     if entry.get("section") not in excluded_sections and entry.get("section") is not None]

    # Filtering out nonsensical witness statements
    witness_statements_f = [
        entry for entry in witness_statements if entry.get("nonsense", 0) < 5]

    # Storing results
    contradictions = []

    for witness_chunk in witness_statements_f:
        witness_text = witness_chunk.get("text")

        for entry in filtered_docs:
            doc_text = entry.get("text")
            response = check_contradiction(witness_text, doc_text)

            result_entry = {
                "witness_statement": witness_text,
                "witness_page": witness_chunk.get("page"),
                "witness_section": witness_chunk.get("section"),
                "doc_text": doc_text,
                "doc_page": entry.get("page"),
                "doc_section": entry.get("section"),
                "result": response
            }

            if "type" in witness_chunk:
                result_entry["w_s_type"] = witness_chunk["type"]
            if "type" in entry:
                result_entry["doc_type"] = entry["type"]

            contradictions.append(result_entry)

    # Saving results to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(contradictions, f, indent=4, ensure_ascii=False)

    print(f"Witness contradictions JSON file created at: {output_path}")


# Loading the input data
with open("../results/combined_doc_sections_with_tables.json", "r", encoding="utf-8") as f:
    document_pages = json.load(f)

with open("../results/w_s_with_tables.json", "r", encoding="utf-8") as f:
    witness_statements = json.load(f)

# Defining the sections to exclude
excluded_sections = {
    "Claimant Exhibit C 5", "Claimant Exhibit C 8",
    "Respondent Exhibit R 1", "Respondent Exhibit R 2",
    "Respondent Exhibit R 3", "Respondent Exhibit R 4",
    "Procedural Order No. 1"
}

# Setting output file path
# output_file = "../results/comp_results1.json"
output_file = "../results/comp_results2.json"

# Running the contradiction detection
detect_witness_contradictions(
    document_pages, witness_statements, excluded_sections, output_file)

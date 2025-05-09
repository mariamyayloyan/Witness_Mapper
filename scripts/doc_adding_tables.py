"""
This script processes a folder of PDF files, extracts tables using Google's Gemini API,
and converts them into descriptive paragraphs using Azure OpenAI GPT-4o.

Environment Variables:
- PROJECT_ID: GCP project ID for Gemini.
- LOCATION_GEMINI: GCP location for the Gemini model.
- AZURE_OPENAI_API_KEY: Azure API key for OpenAI.
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL.

Input:
- Folder of PDFs with names in the format '<section_number> - <section_name>.pdf'
- Existing JSON file: combined_doc_sections.json

Output:
- Updated JSON file with extracted table paragraphs: combined_doc_sections_with_tables.json

Dependencies:
- os
- json
- pathlib
- re
- google (Gemini via `google.genai`)
- openai (Azure OpenAI via `openai.AzureOpenAI`)


"""

import os
import json
import pathlib
import re
from google import genai
from google.genai import types
from openai import AzureOpenAI


def extract_tables_and_add_paragraphs(pdf_path, existing_json):
    """
    Extracts tables from a given PDF file using Gemini 2.5 Flash, then converts each table into a descriptive 
    paragraph using GPT-4o (Azure OpenAI), and appends the results to an existing JSON structure.

    Parameters:
    -----------
    pdf_path : pathlib.Path
        Path to the PDF file to be processed.

    existing_json : list
        A list of dictionaries containing previously extracted content.

    Returns:
    --------
    list
        The updated list with newly added table-based descriptive paragraphs.
    """

    section_name = pdf_path.stem.split(" - ", 1)[-1].strip()

    client_genai = genai.Client(
        vertexai=True,
        project=os.getenv("PROJECT_ID"),
        location=os.getenv("LOCATION_GEMINI")
    )

    # Reading the PDF
    pdf_file = pathlib.Path(pdf_path)
    response = client_genai.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=[
            """Extract only the tables from the attached PDF and return the result in **valid JSON format**.

            Group all tables by their page number. For each entry in the json, use the following structure:

            {
              "page": <page_number>,
              "tables": ...
            }

            If you see any subcategories or subheaders include those.
            Do not include any text or commentary outside the JSON structure. Only return JSON.""",
            types.Part.from_bytes(data=pdf_file.read_bytes(),
                                  mime_type="application/pdf"),
        ],
    )

    # Extracting JSON block from the GenAI response
    def extract_json_code_block(text):
        """
        Extracts and parses a JSON code block from a string that contains a markdown-style JSON block (OpenAI's response).

        Parameters:
        -----------
        text : str
            The string containing the JSON code block.

        Returns:
        --------
        list or None
            A list of dictionaries parsed from the JSON block, or None if parsing fails.
        """
        match = re.search(r"```json\s*(\[\s*{.*?}\s*])\s*```", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                return parsed
            except json.JSONDecodeError:
                print("Failed to parse JSON from extracted block.")
        return None

    tables = extract_json_code_block(response.text)

    # If nothing extracted or no non-empty tables, skip
    if not tables or all(not page.get("tables") for page in tables):
        print(f"No tables found in {pdf_path.name}, skipping.")
        return existing_json

    # Setting up Azure OpenAI client
    client_openai = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    # Processing tables into paragraphs
    for table in tables:
        prompt = f"""
You are given a list of extracted tables from a document. Each entry includes the table data and the page number.

Analyze the table and convert the data into a descriptive paragraph that is clear and understandable and contains all the data provided in the table, ensuring to include all the row, column, subrow, and subcolumn names in the paragraph. Don't include the page number.

Here is the input data:
{json.dumps([table], indent=2)}
"""

        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that turns tables into structured paragraphs."},
                {"role": "user", "content": prompt},
            ]
        )

        generated_paragraph = response.choices[0].message.content.strip()

        # Storing the results in the existing JSON
        existing_json.append({
            "page": table["page"],
            "section": section_name,
            "text": generated_paragraph,
            "type": "table"
        })

    return existing_json


# Loading existing JSON
with open("../results/combined_doc_sections.json", "r", encoding="utf-8") as f:
    existing_json = json.load(f)

# Defining path to the PDF files
pdf_folder = pathlib.Path("../data/32nd-Vis-Moot_Problem_incl_PO2")

# Looping through all PDF files in the folder
for pdf_file in pdf_folder.glob("*.pdf"):
    updated_json = extract_tables_and_add_paragraphs(pdf_file, existing_json)

# Saving the updated JSON to a new file
output_path = "../results/combined_doc_sections_with_tables.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(updated_json, f, indent=4, ensure_ascii=False)

print("Table extraction and updates complete!")

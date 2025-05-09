"""
This script processes a PDF file containing a Witness Statement by:
1. Extracting tables using Google Gemini API and summarizing them using Azure OpenAI GPT-4o.
2. Extracting text using Google Document AI and structuring it into sentence-level JSON using GPT-4o.
3. Combining both outputs and saving them into a JSON file.

Environment Variables:
- PROJECT_ID: GCP project ID
- PROCESSOR_ID: Document AI processor ID
- LOCATION: GCP location for Document AI
- LOCATION_GEMINI: GCP location for Gemini
- AZURE_OPENAI_API_KEY: Azure API key for OpenAI
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL

Dependencies:
- google-cloud-documentai
- google-generativeai
- openai
- pathlib, json, os, re
"""

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1
import os
from openai import AzureOpenAI
import json
import re
from google import genai
from google.genai import types
import pathlib
import os

project_id = os.getenv("PROJECT_ID")
processor_id = os.getenv("PROCESSOR_ID")
location = os.getenv("LOCATION")
location_gemini = os.getenv("LOCATION_GEMINI")
file_path = "../data/32nd-Vis-Moot_Problem_incl_PO2/07 - Claimant Exhibit C 5.pdf"

client_openai = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Extracting Tables


def extract_tables_and_summarize_w_s(file_path, project_id, location_gemini):
    """
    Extracts tables from the given PDF file using Gemini and summarizes them
    into descriptive sentences using Azure OpenAI GPT-4o.

    Args:
        file_path (str): Path to the PDF file.
        project_id (str): GCP project ID.
        location_gemini (str): Location of Gemini model.

    Returns:
        list: A list of JSON dictionaries containing structured sentences per table.
    """
    # Setting up Google GenAI client
    client_genai = genai.Client(
        vertexai=True,
        project=project_id,
        location=location_gemini
    )

    # Reading the PDF
    pdf_file = pathlib.Path(file_path)
    section_name = pdf_file.stem.split(" - ", 1)[-1].strip()
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
    match = re.search(r"```json\s*(.*?)\s*```", response.text, re.DOTALL)
    if not match:
        raise ValueError("Could not extract JSON from GenAI response.")

    tables_data = json.loads(match.group(1))

    # Processing tables into descriptive sentences
    all_sentences = []
    for table in tables_data:
        prompt = f"""
You are given a list of extracted tables from a document. Each entry includes the table data and the page number.

Analyze the table and convert the data into a series of clear, descriptive sentences that can also be clearly understood taken by itself, make sure to include all the row, column, subrow, subcolumn names in them. For each sentence, include:
- "text": the sentence itself (don't include a page number in the sentence),
- "page": the page number it comes from.

Your response should be a valid JSON array of dictionaries in this format:
[
  {{
    "section": Witness Statement ({section_name})
    "text": "...",
    "page": {table["page"]},
    "type": "table",
    "nonsense": A nonsense score from 0 to 10 (10 as a sentence that doesn't contain a fact or a useful statement and 0 as a sentence that contains very useful information), give a higher score to incomplete sentences.
  }},
  ...
]

Only output valid JSON. Do not include any extra explanation or commentary.

Here is the input data:
{json.dumps([table], indent=2)}
"""
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that turns tables into structured paragraph sentences in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()
        try:
            result = json.loads(content)
            all_sentences.extend(result)
        except json.JSONDecodeError:
            print(
                f"Failed to parse JSON from table on page {table.get('page')}. Skipping...")
            print(content)

    return all_sentences


result_tables = extract_tables_and_summarize_w_s(
    file_path, project_id, location_gemini)


# Extracting Text

def extract_sentences_from_w_s(file_path, project_id, location, processor_id):
    """
    Extracts textual sentences from the PDF using Google Document AI and formats them
    as structured JSON entries via GPT-4o, including section and nonsense score.

    Args:
        file_path (str): Path to the PDF file.
        project_id (str): GCP project ID.
        location (str): GCP location for Document AI.
        processor_id (str): Processor ID for Document AI.

    Returns:
        list: A list of dictionaries with extracted structured text per sentence.
    """
    # Initializing Document AI client
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client_doc_ai = documentai_v1.DocumentProcessorServiceClient(
        client_options=opts)
    full_processor_name = client_doc_ai.processor_path(
        project_id, location, processor_id)

    # Getting processor
    request = documentai_v1.GetProcessorRequest(name=full_processor_name)
    processor = client_doc_ai.get_processor(request=request)

    # Reading and processing file
    with open(file_path, "rb") as image:
        image_content = image.read()

    raw_document = documentai_v1.RawDocument(
        content=image_content, mime_type="application/pdf",)
    request = documentai_v1.ProcessRequest(
        name=processor.name, raw_document=raw_document)
    result = client_doc_ai.process_document(request=request)
    document = result.document
    text = document.text

    # Extracting page-wise text
    page_texts = []
    for page in document.pages:
        page_text = ""
        for segment in page.layout.text_anchor.text_segments:
            start = int(segment.start_index) if segment.start_index else 0
            end = int(segment.end_index)
            page_text += text[start:end]
        page_texts.append(page_text.strip())

    # Function to add overlaps
    def add_overlaps(chunks, overlap_words=15):
        overlapped_chunks = []
        for i in range(len(chunks)):
            current_chunk = chunks[i]
            current_words = current_chunk.split()
            if i > 0:
                prev_words = chunks[i - 1].split()
                overlap = prev_words[-overlap_words:] if len(
                    prev_words) >= overlap_words else prev_words
                current_chunk = " ".join(overlap) + \
                    " " + " ".join(current_words)
            overlapped_chunks.append(current_chunk)
        return overlapped_chunks

    overlapped_chunks = add_overlaps(page_texts, overlap_words=45)

    all_results = []
    pdf_file = pathlib.Path(file_path)
    section_name = pdf_file.stem.split(" - ", 1)[-1].strip()

    # Processing each chunk
    for i, chunk in enumerate(overlapped_chunks):
        page_number = i + 1
        prompt = f"""
Break down the following text sentence by sentence into a structured JSON format. Each sentence should include:
1. Section title "Witness Statement ({section_name})" under "section".
2. {page_number} under "page".
3. Sentence text under "text" (ignore footers).
4. A nonsense score from 0 to 10 (10 as a sentence that doesn't contain a fact or a useful statement and 0 as a sentence that contains very useful information) under "nonsense", give a higher score to incomplete sentences (sentences with no ending or no beginning).
Exclude things you think may be a table text.

Here is the text to analyze:

{chunk}
"""
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that converts legal text into structured JSON, focusing on clarity and completeness."},
                {"role": "user", "content": prompt}
            ]
        )

        result_text = response.choices[0].message.content.strip()
        match = re.search(r"```json\s*(.*?)\s*```", result_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                parsed = json.loads(json_str)
                all_results.extend(parsed)
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed on page {page_number}: {e}")
                print(json_str)
        else:
            raise ValueError("Could not extract JSON from GenAI response.")

    return all_results


result_text = extract_sentences_from_w_s(
    file_path, project_id, location, processor_id)

# Combining text and tables from the Witness Statement
result_w_s = result_text + result_tables

# Storing in a JSON
output_path = "../results/w_s_with_tables.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result_w_s, f, indent=4, ensure_ascii=False)

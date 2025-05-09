"""
This script processes a folder of split arbitration PDFs using Google Cloud Document AI OCR
and extracts text content from each page. The output is a structured JSON file containing 
page-wise text along with the section name and page number.

In real arbitration scenarios, each document—such as procedural orders, claims, and exhibits—
is shared separately. This script assumes that the PDF files have already been split by section 
(e.g., using another script) and stored in a directory.

The Document AI processor is configured via environment variables:
- PROJECT_ID
- PROCESSOR_ID
- LOCATION

Each PDF is passed to Document AI, which performs OCR and returns structured text. The results 
are saved in `../results/combined_doc_sections.json` .

Dependencies:
- Google Cloud Document AI Python SDK
- os, json, re

"""

import os
import json
import re
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1

os.makedirs("../results", exist_ok=True)

# Loading environment variables
project_id = os.getenv("PROJECT_ID")
processor_id = os.getenv("PROCESSOR_ID")
location = os.getenv("LOCATION")

# Initializing Document AI client
opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
client_doc_ai = documentai_v1.DocumentProcessorServiceClient(
    client_options=opts)
full_processor_name = client_doc_ai.processor_path(
    project_id, location, processor_id)


def extract_text_per_page(pdf_path, section_name):
    """Processes a PDF using Document AI OCR and returns a list of {page, section, text} dictionaries"""
    with open(pdf_path, "rb") as f:
        image_content = f.read()

    raw_document = documentai_v1.RawDocument(
        content=image_content,
        mime_type="application/pdf",
    )

    request = documentai_v1.ProcessRequest(
        name=full_processor_name, raw_document=raw_document)
    result = client_doc_ai.process_document(request=request)
    document = result.document

    text = document.text
    page_texts = []

    for i, page in enumerate(document.pages):
        page_text = ""
        for segment in page.layout.text_anchor.text_segments:
            start = int(segment.start_index) if segment.start_index else 0
            end = int(segment.end_index)
            page_text += text[start:end]

        page_texts.append({
            "page": i + 1,
            "section": section_name,
            "text": page_text.strip()
        })

    return page_texts


# Folder containing split PDFs
pdf_folder = "../data/32nd-Vis-Moot_Problem_incl_PO2"
combined_data = []

# Processing each PDF in the folder
for filename in sorted(os.listdir(pdf_folder)):
    if filename.endswith(".pdf"):
        section_match = re.match(r"\d+\s*-\s*(.+)\.pdf", filename)
        section = section_match.group(
            1) if section_match else filename.replace(".pdf", "")
        pdf_path = os.path.join(pdf_folder, filename)

        print(f"Processing: {filename}")
        section_pages = extract_text_per_page(pdf_path, section)
        combined_data.extend(section_pages)

# Saving to JSON
output_path = "../results/combined_doc_sections.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(combined_data, f, indent=4, ensure_ascii=False)

print(f"All PDFs processed and combined into '{output_path}'")

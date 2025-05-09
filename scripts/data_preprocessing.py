"""
This script splits a single compiled arbitration case PDF into separate documents based on defined section titles 
and their corresponding start pages.

In real arbitration scenarios, each document—such as exhibits, procedural orders, emails, etc. is typically 
shared as a standalone file. This script replicates that practice for easier document management, review, and referencing.

The sections are defined in a dictionary where each key is a section name and each value is the 1-indexed starting page 
of that section in the original PDF. The resulting PDFs are saved to the specified output folder with filenames indicating 
their order and section name.

Dependencies:
- PyMuPDF (imported as `fitz`)
- os module for file operations

"""

import fitz
import os


def split_pdf_by_sections(input_path, output_folder, sections):
    """
    Splits a PDF file into multiple smaller PDFs based on section titles and start pages.

    Parameters:
    -----------
    input_path : str
        The file path to the input PDF.
    output_folder : str
        The directory where the split PDF sections will be saved.
    sections : dict
        A dictionary where keys are section titles and values are the starting page numbers (1-indexed) 
        for each section in the input PDF.

    The function creates individual PDF files for each section and saves them in the output folder 
    with filenames prefixed by their order.
    """
    # Opening the original PDF
    doc = fitz.open(input_path)

    # Ensuring output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Converting section dictionary to a list for ordered processing
    section_items = list(sections.items())

    for i in range(len(section_items)):
        title, start_page = section_items[i]
        start = start_page - 1

        # Determining end page
        if i < len(section_items) - 1:
            end = section_items[i + 1][1] - 1
        else:
            end = len(doc)

        # Extracting and saving section
        subdoc = fitz.open()
        subdoc.insert_pdf(doc, from_page=start, to_page=end - 1)

        filename = f"{i+1:02d} - {title}.pdf"
        subdoc.save(os.path.join(output_folder, filename))
        subdoc.close()

    doc.close()
    print("PDF successfully split.")


# File paths
input_path = "../data/32nd-Vis-Moot_Problem_incl_PO2.pdf"
output_folder = "../data/32nd-Vis-Moot_Problem_incl_PO2"


# Sections based on table of content
sections = {
    "Letter by Langweiler": 3,
    "Request for Arbitration": 4,
    "Claimant Exhibit C 1": 10,
    "Claimant Exhibit C 2": 12,
    "Claimant Exhibit C 3": 16,
    "Claimant Exhibit C 4": 17,
    "Claimant Exhibit C 5": 18,
    "Claimant Exhibit C 6": 21,
    "Claimant Exhibit C 7": 22,
    "Letters by FAI": 23,
    "Letter by Fasttrack": 26,
    "Answer to the Request for Arbitration": 27,
    "Respondent Exhibit R 1": 31,
    "Respondent Exhibit R 2": 33,
    "Respondent Exhibit R 3": 34,
    "Respondent Exhibit R 4": 35,
    "Letter by Langweiler Objecting to Admittance of Document": 36,
    "Claimant Exhibit C 8": 38,
    "Letters by FAI (2)": 39,
    "Letter by FAI Concerning the Decisions Made by the Board": 41,
    "Letter by FAI Confirming the Party-nominated Arbitrators": 43,
    "Letters by FAI Concerning the Appointment of Presiding Arbitrator": 44,
    "Letters by Greenhouse": 50,
    "Procedural Order No. 1": 52,
    "Procedural Order No. 2": 54
}

split_pdf_by_sections(
    input_path,
    output_folder,
    sections=sections
)

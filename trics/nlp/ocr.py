import os 
from typing import List
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader, PdfWriter
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
import time


def get_text(doc: str, first_page: int = 1, last_page: int = 2) -> str:
    """
    Extracts text from specified pages of a PDF document using OCR.

    Args:
    doc (str): The file path to the PDF document.
    first_page (int): The first page to start converting for text extraction.
    last_page (int): The last page to convert for text extraction.

    Returns:
    str: The extracted text from the specified range of pages of the PDF.
    """
    # Initialize an empty string to hold all the text
    text = ''

    # Convert the pdf pages to images
    images = convert_from_path(doc, first_page=first_page, last_page=last_page)

    for image in images:
        # Convert each image to text
        text += pytesseract.image_to_string(image, lang='eng')

    return text

def cleanup_temp_file(temp_filename):
    """Remove the temporary file if it exists."""
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

def extract_pages(source_file, num_pages):
    """Extracts the first 'num_pages' from the source PDF file and saves it to a temporary file."""
    reader = PdfReader(source_file)
    pdf_writer = PdfWriter()

    for page in range(min(num_pages, len(reader.pages))):
        pdf_writer.add_page(reader.pages[page])

    with open('temp.pdf', 'wb') as temp_pdf:
        pdf_writer.write(temp_pdf)

def read_pdf_with_azure(temp_filename, computervision_client):
    """Uses Azure Computer Vision to read text from a PDF stored in 'temp_filename'."""
    try:
        with open(temp_filename, "rb") as pdf:
            read_response = computervision_client.read_in_stream(pdf, raw=True)
        operation_location_remote = read_response.headers["Operation-Location"]
        operation_id = operation_location_remote.split("/")[-1]

        while True:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)

        if read_result.status == OperationStatusCodes.succeeded:
            text = ""
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    text += line.text + "\n"
            return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def extract_text(DocketNo, file_pdf, k, data_folder, computervision_client):
    temp_filename = 'temp.pdf'
    cleanup_temp_file(temp_filename)
    
    """Coordinates the extraction of text from a PDF using Azure Computer Vision."""
    source_file = os.path.join(data_folder, DocketNo, file_pdf)
    
    extract_pages(source_file, k)
    text = read_pdf_with_azure(temp_filename, computervision_client)
    cleanup_temp_file(temp_filename)  # Optional: Clean up again after reading

    return text



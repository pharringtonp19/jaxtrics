import os 
from typing import List
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

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


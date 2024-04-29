import os 
from typing import List
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader, PdfWriter
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from typing import Optional
import time

def extract_first_k_pages(source_file: str, temp_file: str, num_pages: int) -> None:
    """
    Extracts the first 'k' pages from the source PDF file and saves it to a temporary file.

    Args:
    source_file (str): The file path to the source PDF document.
    temp_file (str): The file path where the temporary PDF should be saved.
    num_pages (int): The number of pages to extract from the source PDF.

    Returns:
    None
    """
    reader = PdfReader(source_file)
    pdf_writer = PdfWriter()

    for page in range(min(num_pages, len(reader.pages))):
        pdf_writer.add_page(reader.pages[page])

    with open(temp_file, 'wb') as temp_pdf:
        pdf_writer.write(temp_pdf)




def read_pdf_with_azure(temp_filename: str, computervision_client: ComputerVisionClient) -> Optional[str]:
    """
    Uses Azure Computer Vision to read text from a PDF stored in 'temp_filename'.

    Args:
    temp_filename (str): The path to the temporary PDF file.
    computervision_client (ComputerVisionClient): An instance of Azure's Computer Vision client.

    Returns:
    Optional[str]: The extracted text if the process succeeds, otherwise None.
    """
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
        return None


def extract_text(data_folder, temp_file, DocketNo, file_pdf, intitial_k_pages, computervision_client):

    # Ensure there is no temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # Define File Path
    source_file = os.path.join(data_folder, DocketNo, file_pdf)

    # Write Extracted Pages to a temp file
    extract_first_k_pages(source_file, temp_file, intitial_k_pages)

    # Extract Text from the `initial_k_pages`
    text = read_pdf_with_azure(temp_filename, computervision_client)

    return text

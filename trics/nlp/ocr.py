import os 
from typing import List
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader, PdfWriter
import base64
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from typing import Optional
import time
import base64
# from azure.cognitiveservices.vision.computervision import ComputerVisionClient
# from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes

def extract_first_k_pages(source_file: str, temp_file: str, num_pages: int) -> None:
    """
    Extracts the first 'num_pages' pages from the source PDF file and saves it to a temporary file.

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


def read_pdf_with_azure(client: DocumentIntelligenceClient, temp_filename: str) -> str:
    with open(temp_filename, 'rb') as f:
        pdf_bytes = f.read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

    # Pass the base64 encoded data
    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        analyze_request={"base64Source": pdf_base64}
    )
    result = poller.result()  # result is of type AnalyzeResult

    return result.content  # Assuming result.content is of type str

def extract_text(client: DocumentIntelligenceClient, file_folder: str, temp_file: str, file_name: str, num_pages: int, DocketNo: str) -> Optional[str]:
    """
    Extracts text from the initial pages of a specified PDF file using Azure's Computer Vision.

    Args:
    client (DocumentIntelligenceClient): An instance of Azure's Document Intelligence client to use for OCR.
    file_folder (str): The path to the base directory where PDF files are stored.
    temp_file (str): The file path for the temporary file used to store extracted pages.
    file_name (str): The name of the PDF file from which text is to be extracted.
    initial_k_pages (int): The number of initial pages from the PDF to extract and analyze.
    DocketNo (str): The docket number associated with the file, used to construct the file path.

    Returns:
    Optional[str]: The extracted text from the PDF if successful, None otherwise.
    """
    # Ensure there is no temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # Define File Path
    source_file = os.path.join(file_folder, DocketNo, file_name)

    # Write Extracted Pages to a temp file
    extract_first_k_pages(source_file, temp_file, num_pages)

    # Extract Text from the `initial_k_pages`
    text = read_pdf_with_azure(client, temp_file)

    return text

# def read_pdf_with_azure(temp_filename: str, computervision_client: ComputerVisionClient) -> Optional[str]:
#     """
#     Uses Azure Computer Vision to read text from a PDF stored in 'temp_filename'.

#     Args:
#     temp_filename (str): The path to the temporary PDF file.
#     computervision_client (ComputerVisionClient): An instance of Azure's Computer Vision client.

#     Returns:
#     Optional[str]: The extracted text if the process succeeds, otherwise None.
#     """
#     try:
#         with open(temp_filename, "rb") as pdf:
#             read_response = computervision_client.read_in_stream(pdf, raw=True)
#         operation_location_remote = read_response.headers["Operation-Location"]
#         operation_id = operation_location_remote.split("/")[-1]

#         while True:
#             read_result = computervision_client.get_read_result(operation_id)
#             if read_result.status not in ['notStarted', 'running']:
#                 break
#             time.sleep(1)

#         if read_result.status == OperationStatusCodes.succeeded:
#             text = ""
#             for text_result in read_result.analyze_result.read_results:
#                 for line in text_result.lines:
#                     text += line.text + "\n"
#             return text
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None

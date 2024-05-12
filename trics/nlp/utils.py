import os
from typing import List
import csv
from IPython.display import display
from IPython.display import Markdown
import textwrap
import logging
import PyPDF2
import numpy as np 


def create_csv_with_headers(filename: str, headers: List[str]) -> None:
    """
    Creates a CSV file with the specified headers if the file does not exist.

    Args:
    filename (str): The path to the CSV file to be created.
    headers (List[str]): A list of header names to write to the CSV file.
    """
    print(f"File not found. Creating new file with headers: {headers}")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

def check_file_title(title: str, word: str) -> bool:
    """
    Checks if a word is present in the title, case-insensitively.

    Args:
    title (str): The title in which to search for the word.
    word (str): The word to search for in the title.

    Returns:
    bool: True if the word is found in the title, otherwise False.
    """
    if word.lower() in title.lower():
        return True
    else:
        return False

def write_to_csv(file: str, case: str, summary: List[str]) -> None:
    """
    Appends a row to a CSV file, consisting of a case identifier followed by a summary.

    Args:
    file (str): Path to the CSV file where the data should be appended.
    case (str): The case identifier to be written.
    summary (List[str]): A list of summary elements to be written alongside the case.
    """
    with open(file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([case] + summary)

def find_directories_with_specific_pdf(base_folder: str, pdf_title: str) -> List[str]:
    """
    Find all directories within the base folder that contain at least one PDF file with a specific title.

    Args:
    base_folder (str): The path to the base directory to search within.
    pdf_title (str): The title of the PDF file to search for (including '.pdf' suffix).

    Returns:
    list: A list of directory names containing the specified PDF file.
    """
    directories_with_specific_pdfs = []
    for entry in os.listdir(base_folder):
        dir_path = os.path.join(base_folder, entry)
        if os.path.isdir(dir_path):
            files_in_dir = os.listdir(dir_path)
            if any(file_name == pdf_title for file_name in files_in_dir):
                directories_with_specific_pdfs.append(entry)

    return directories_with_specific_pdfs

def average_pages_in_pdfs(base_folder: str, pdf_title: str):
    pages = []
    for entry in os.listdir(base_folder):
        dir_path = os.path.join(base_folder, entry)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename == pdf_title:
                    pdf_path = os.path.join(dir_path, filename)
                    try:
                        with open(pdf_path, 'rb') as file:
                            reader = PyPDF2.PdfReader(file)
                            pages.append(len(reader.pages))
                    except FileNotFoundError:
                        logging.error(f"File not found: {pdf_path}")
                    except Exception as e:
                        logging.error(f"Failed to read {filename}: {str(e)}")
    
    return np.array(pages)

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def blank_out_addresses_and_zip_codes(text):
    """
    Replace all addresses and 5-digit zip codes in the given text with '[ADDRESS]' and 'XXXXX' respectively.
    
    Parameters:
    text (str): The input text where addresses and zip codes need to be blanked out.
    
    Returns:
    str: The modified text with addresses and zip codes replaced.
    """
    # Regular expression pattern to match simple addresses
    address_pattern = r'\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive)\b'
    
    # Replace found addresses with '[ADDRESS]'
    modified_text = re.sub(address_pattern, '[ADDRESS]', text)
    
    # Regular expression pattern to match 5-digit zip codes
    zip_code_pattern = r'\b\d{5}\b'
    
    # Replace all occurrences of zip code pattern with 'XXXXX'
    modified_text = re.sub(zip_code_pattern, 'XXXXX', modified_text)
    
    return modified_text

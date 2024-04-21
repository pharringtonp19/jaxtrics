from typing import List
import csv

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

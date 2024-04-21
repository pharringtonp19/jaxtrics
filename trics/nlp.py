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

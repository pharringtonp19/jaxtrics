import numpy as np
import re

def batch_matrix(matrix: np.array, zip_codes: np.array):
    # Create batches as before
    Batches = {}
    Masks = {}
    unique_zip_codes = np.unique(zip_codes)
    for zip_code in unique_zip_codes:
        indices = np.where(zip_codes == zip_code)[0]
        batch = matrix[indices, :]
        Batches[zip_code] = batch

    # Identify maximum size
    max_rows = max(batch.shape[0] for batch in Batches.values())

    # Pad batches and create masks
    for zip_code, batch in Batches.items():
        padding_rows = max_rows - batch.shape[0]
        
        # Add padding
        padded_batch = np.pad(batch, pad_width=((0, padding_rows), (0, 0)), mode='constant')
        
        # Create mask where 1 indicates original data and 0 indicates padding
        mask = np.pad(np.ones(shape=(batch.shape[0], 1)),pad_width=((0, padding_rows), (0, 0)), mode='constant')
        
        Batches[zip_code] = padded_batch
        Masks[zip_code] = mask
    
    # Convert the dictionary values to a list of matrices
    return np.stack(list(Batches.values())), np.stack(list(Masks.values()))


def single_unbatch_matrix(batch_X: np.array, mask: np.array):
    """
    Removes padding from batch matrices based on a given mask.

    Args:
    batch_X (np.array): The first batch matrix with padding.
    mask (np.array): A mask indicating the original data (non-padded parts).

    Returns:
    tuple: A single array with padding removed.
    """

    # Validate mask dimensions
    if mask.ndim != batch_X.ndim:
        raise ValueError("Mask and input batch dimensions do not match")

    # Use broadcasting to apply mask and select non-zero rows in one step
    unpad_X = batch_X[mask.astype(bool).reshape(-1)]

    return unpad_X

def unbatch_matrix(X: np.array,masks: np.array) -> dict:
    original_X = {}

    for i in range(X.shape[0]):
        original_X[i]  = single_unbatch_matrix(X[i], masks[i]).reshape(-1, X.shape[-1])
    
    return np.vstack((list(original_X.values()))) 


def find_first_yes_or_no(text):
    # This regular expression pattern looks for 'Yes' or 'No' (case insensitive)
    pattern = r'\b(Yes|No)\b'
    
    # Search the text for the pattern
    match = re.search(pattern, text, re.IGNORECASE)
    
    # If a match is found, return the matched text, otherwise return None
    return match.group(0) if match else None

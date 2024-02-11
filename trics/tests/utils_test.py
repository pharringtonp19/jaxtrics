import numpy as np
from trics.utils import batch_matrix, unbatch_matrix

def batch_matrix_with_padding(matrix: np.array, zip_codes: np.array) -> dict:
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

def test_batch_matrix_with_padding():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    zip_codes = np.array([1, 2, 1])
    batched_matrix, mask = batch_matrix(matrix, zip_codes)
    assert batched_matrix.shape == (2, 2, 3)
    assert mask.shape == (2, 2, 1)


def test_unbatch_matrix():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    zip_codes = np.array([1, 2, 1])
    batched_matrix, mask = batch_matrix(matrix, zip_codes)
    matrix_processed = unbatch_matrix(batched_matrix, mask)
    assert matrix_processed.shape == matrix.shape, matrix_processed 
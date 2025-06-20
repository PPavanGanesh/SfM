import numpy as np

def EssentialMatrixFromFundamentalMatrix(F, K):
    """
    Calculate the essential matrix from the fundamental matrix.
    
    Args:
        F: 3x3 fundamental matrix
        K: 3x3 camera intrinsic matrix
        
    Returns:
        E: 3x3 essential matrix
    """
    # E = K^T * F * K
    E = K.T @ F @ K
    
    # Enforce the property that E has two equal singular values and one zero
    U, S, Vt = np.linalg.svd(E)
    
    # Set the two non-zero singular values to 1
    S = np.array([1.0, 1.0, 0.0])
    
    # Reconstruct E with the corrected singular values
    E = U @ np.diag(S) @ Vt
    
    return E

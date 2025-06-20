import numpy as np

def ExtractCameraPose(E):
    """
    Extract the four possible camera poses from the essential matrix.
    
    Args:
        E: 3x3 essential matrix
        
    Returns:
        poses: List of four possible camera poses, each as a tuple (C, R)
            where C is the camera center and R is the rotation matrix
    """
    # Perform SVD on the essential matrix
    U, S, Vt = np.linalg.svd(E)
    
    # Ensure proper rotation matrix (det(R) = 1)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt
    
    # Define the W matrix as given in the project document
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    
    # Extract rotation matrices
    R1 = U @ W @ Vt
    R2 = U @ W @ Vt  # Same as R1
    R3 = U @ W.T @ Vt
    R4 = U @ W.T @ Vt  # Same as R3
    
    # Extract translation vectors (camera centers)
    C1 = U[:, 2]  # Third column of U
    C2 = -U[:, 2]  # Negative of third column of U
    C3 = U[:, 2]  # Same as C1
    C4 = -U[:, 2]  # Same as C2
    
    # Check if the rotations are proper (det(R) = 1)
    # If not, correct them
    if np.linalg.det(R1) < 0:
        R1 = -R1
        C1 = -C1
    
    if np.linalg.det(R2) < 0:
        R2 = -R2
        C2 = -C2
        
    if np.linalg.det(R3) < 0:
        R3 = -R3
        C3 = -C3
        
    if np.linalg.det(R4) < 0:
        R4 = -R4
        C4 = -C4
    
    # Create the four possible camera poses
    poses = [
        (C1, R1),
        (C2, R2),
        (C3, R3),
        (C4, R4)
    ]
    
    return poses

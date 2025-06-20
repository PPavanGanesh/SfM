import numpy as np

def normalize(points):
    """
    Normalize points to improve numerical stability.
    
    Args:
        points: Nx2 array of points
        
    Returns:
        normalized_points: Nx3 array of normalized homogeneous points
        T: 3x3 transformation matrix
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Shift points to have centroid at origin
    shifted_points = points - centroid
    
    # Compute average distance from origin
    avg_dist = np.mean(np.sqrt(np.sum(shifted_points**2, axis=1)))
    
    # Scale factor to make average distance sqrt(2)
    scale = np.sqrt(2) / avg_dist if avg_dist > 0 else 1.0
    
    # Create transformation matrix
    T = np.array([
        [scale, 0, -scale*centroid[0]],
        [0, scale, -scale*centroid[1]],
        [0, 0, 1]
    ])
    
    # Convert points to homogeneous coordinates and apply transformation
    homogeneous_points = np.column_stack((points, np.ones(len(points))))
    normalized_points = (T @ homogeneous_points.T).T
    
    return normalized_points, T

def EstimateFundamentalMatrix(pts1, pts2):
    """
    Estimate the fundamental matrix from corresponding points using the normalized 8-point algorithm.
    
    Args:
        pts1: Nx2 array of points in the first image
        pts2: Nx2 array of corresponding points in the second image
        
    Returns:
        F: 3x3 fundamental matrix with rank 2
    """
    # Check if we have at least 8 points
    if pts1.shape[0] < 8:
        return None
    
    # Normalize points
    pts1_norm, T1 = normalize(pts1)
    pts2_norm, T2 = normalize(pts2)
    
    # Construct the constraint matrix A
    A = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        x1, y1 = pts1_norm[i, 0], pts1_norm[i, 1]
        x2, y2 = pts2_norm[i, 0], pts2_norm[i, 1]
        A[i] = [x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Solve for F using SVD
    _, _, V = np.linalg.svd(A, full_matrices=True)
    F = V[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # Set the smallest singular value to zero
    F = U @ np.diag(S) @ Vt
    
    # Denormalize
    F = T2.T @ F @ T1
    
    # Normalize F
    F = F / np.linalg.norm(F)

    if F[2, 2] < 0:
        F = -F
    
    return F

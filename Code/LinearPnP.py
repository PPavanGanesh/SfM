import numpy as np

def LinearPnP(X, x, K):
    """
    Estimate camera pose using linear PnP.
    
    Args:
        X: Nx3 array of 3D points in world coordinates
        x: Nx2 array of 2D points in image coordinates
        K: 3x3 camera intrinsic matrix
        
    Returns:
        C: 3x1 camera center
        R: 3x3 rotation matrix
    """
    # Number of points
    n = X.shape[0]
    
    if n < 6:
        print("Error: At least 6 points are required for LinearPnP")
        return None, None
    
    # Normalize image points using K inverse
    K_inv = np.linalg.inv(K)
    x_normalized = np.zeros((n, 2))
    
    for i in range(n):
        p = np.array([x[i, 0], x[i, 1], 1.0])
        p_normalized = K_inv @ p
        x_normalized[i, 0] = p_normalized[0] / p_normalized[2]
        x_normalized[i, 1] = p_normalized[1] / p_normalized[2]
    
    # Construct the measurement matrix A
    A = np.zeros((2*n, 12))
    
    for i in range(n):
        X_i = X[i]
        x_i = x_normalized[i, 0]
        y_i = x_normalized[i, 1]
        
        # Fill in the rows of A
        A[2*i] = [X_i[0], X_i[1], X_i[2], 1, 0, 0, 0, 0, -x_i*X_i[0], -x_i*X_i[1], -x_i*X_i[2], -x_i]
        A[2*i+1] = [0, 0, 0, 0, X_i[0], X_i[1], X_i[2], 1, -y_i*X_i[0], -y_i*X_i[1], -y_i*X_i[2], -y_i]
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    
    # Extract rotation and translation
    R = P[:, :3]
    t = P[:, 3]
    
    # Enforce the orthogonality constraint on R
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    
    # Ensure det(R) = 1
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    
    # Calculate camera center C = -R^T * t
    C = -R.T @ t
    
    return C, R

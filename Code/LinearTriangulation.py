import numpy as np

def LinearTriangulation(K, C1, R1, C2, R2, pts1, pts2):
    """
    Triangulate 3D points from 2D correspondences using linear triangulation.
    
    Args:
        K: 3x3 camera intrinsic matrix
        C1: 3x1 camera center of the first camera
        R1: 3x3 rotation matrix of the first camera
        C2: 3x1 camera center of the second camera
        R2: 3x3 rotation matrix of the second camera
        pts1: Nx2 array of points in the first image
        pts2: Nx2 array of corresponding points in the second image
        
    Returns:
        X: Nx3 array of triangulated 3D points
    """
    # Number of point correspondences
    num_points = pts1.shape[0]
    X = np.zeros((num_points, 3))
    
    # Compute projection matrices
    # P = K * R * [I | -C]
    P1 = K @ np.hstack((R1, -R1 @ C1.reshape(3, 1)))
    P2 = K @ np.hstack((R2, -R2 @ C2.reshape(3, 1)))
    
    for i in range(num_points):
        # Get homogeneous coordinates
        x1 = np.append(pts1[i], 1)
        x2 = np.append(pts2[i], 1)
        
        # Construct the A matrix for the linear system
        A = np.zeros((4, 4))
        
        # From first image
        A[0] = x1[0] * P1[2] - P1[0]  # x * P3 - P1
        A[1] = x1[1] * P1[2] - P1[1]  # y * P3 - P2
        
        # From second image
        A[2] = x2[0] * P2[2] - P2[0]  # x' * P3' - P1'
        A[3] = x2[1] * P2[2] - P2[1]  # y' * P3' - P2'
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X_homog = Vt[-1]  # Last row of Vt
        
        # Convert from homogeneous to Euclidean coordinates
        X[i] = X_homog[:3] / X_homog[3]
    
    return X

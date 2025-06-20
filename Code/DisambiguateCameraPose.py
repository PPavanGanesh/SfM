import numpy as np
from LinearTriangulation import LinearTriangulation

def DisambiguateCameraPose(poses, K, pts1, pts2, X_set=None):
    """
    Disambiguate camera pose by checking the cheirality condition.
    
    Args:
        poses: List of four possible camera poses, each as a tuple (C, R)
        K: 3x3 camera intrinsic matrix
        pts1: Nx2 array of points in the first image
        pts2: Nx2 array of corresponding points in the second image
        
    Returns:
        C: 3x1 camera center of the correct pose
        R: 3x3 rotation matrix of the correct pose
        X: Nx3 array of triangulated 3D points
    """
    # First camera is at the origin with identity rotation
    C1 = np.zeros(3)
    R1 = np.eye(3)
    
    max_positive_depths = 0
    best_pose_idx = -1
    best_X = None
    
    for i, (C2, R2) in enumerate(poses):
        # Triangulate points
        X = LinearTriangulation(K, C1, R1, C2, R2, pts1, pts2)
        
        # Ensure X is in Euclidean coordinates (if returned as homogeneous)
        if X.shape[1] == 4:
            X = X[:, :3] / X[:, 3].reshape(-1, 1)
        
        # Count points satisfying cheirality condition for both cameras
        n_positive_depths = 0
        
        for j in range(X.shape[0]):
            # Check first camera: point must be in front of camera
            # For first camera at origin, this is simply Z > 0
            if X[j, 2] > 0:
                # Check second camera: r3(X-C) > 0
                r3 = R2[2, :].reshape(1, -1)  # Third row of R2
                X_j = X[j, :].reshape(-1, 1)
                C2_vec = C2.reshape(-1, 1)
                
                if r3.dot(X_j - C2_vec) > 0:
                    n_positive_depths += 1
        
        if n_positive_depths > max_positive_depths:
            max_positive_depths = n_positive_depths
            best_pose_idx = i
            best_X = X
    
    # Return the best pose and corresponding 3D points
    if best_pose_idx != -1:
        C, R = poses[best_pose_idx]
        return C, R, best_X, best_pose_idx
    else:
        # Default to the first pose if no good pose is found
        C, R = poses[0]
        X = LinearTriangulation(K, C1, R1, C, R, pts1, pts2)
        # Ensure X is in Euclidean coordinates
        if X.shape[1] == 4:
            X = X[:, :3] / X[:, 3].reshape(-1, 1)
        return C, R, X, best_pose_idx #### changes here add the best_pose_idx if not running

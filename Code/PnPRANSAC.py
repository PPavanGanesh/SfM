import numpy as np
from LinearPnP import LinearPnP

def PnPRANSAC(X, x, K, num_iterations=6000, error_threshold=30.0):
    """
    Estimate camera pose using PnP with RANSAC to handle outliers.
    
    Args:
        X: Nx3 array of 3D points in world coordinates
        x: Nx2 array of 2D points in image coordinates
        K: 3x3 camera intrinsic matrix
        num_iterations: Number of RANSAC iterations
        error_threshold: Threshold for reprojection error (in pixels)
        
    Returns:
        C: 3x1 camera center of the best pose
        R: 3x3 rotation matrix of the best pose
        inliers: Indices of inlier correspondences
    """
    # Check if we have enough correspondences
    n = X.shape[0]
    if n < 6:
        print("Error: At least 6 points are required for PnP RANSAC")
        return None, None, []
    
    # Initialize variables to store the best model
    max_inliers = 0
    best_C = None
    best_R = None
    best_inlier_indices = []
    
    for i in range(num_iterations):
        # Randomly select 6 correspondences
        sample_indices = np.random.choice(n, 6, replace=False)
        X_sample = X[sample_indices]
        x_sample = x[sample_indices]
        
        # Estimate camera pose using LinearPnP
        C, R = LinearPnP(X_sample, x_sample, K)
        
        if C is None or R is None:
            continue
        
        # Compute projection matrix P = K[R|-RC]
        P = K @ np.hstack((R, -R @ C.reshape(3, 1)))
        
        # Count inliers by measuring reprojection error
        inlier_indices = []
        for j in range(n):
            # Convert 3D point to homogeneous coordinates
            X_homog = np.append(X[j], 1)
            
            # Project 3D point to image
            x_proj = P @ X_homog
            x_proj = x_proj[:2] / x_proj[2]  # Normalize
            
            # Compute reprojection error
            # error = np.sum((x_proj - x[j])**2)

            # Compute reprojection error
            error = np.sqrt(np.sum((x_proj - x[j])**2))  # Take square root
            
            # Check if the point is an inlier
            if error < error_threshold:
                inlier_indices.append(j)
        
        # Update the best model if we found more inliers
        if len(inlier_indices) > max_inliers:
            max_inliers = len(inlier_indices)
            best_C = C
            best_R = R
            best_inlier_indices = inlier_indices
    
    # If no inliers were found, return the best we have
    if max_inliers == 0:
        print("Warning: No inliers found in PnP RANSAC")
        return None, None, []
    
    # Refine the pose using all inliers
    if len(best_inlier_indices) >= 6:
        X_inliers = X[best_inlier_indices]
        x_inliers = x[best_inlier_indices]
        C_refined, R_refined = LinearPnP(X_inliers, x_inliers, K)
        if C_refined is not None and R_refined is not None:
            best_C = C_refined
            best_R = R_refined
    
    print(f"PnP RANSAC found {max_inliers} inliers out of {n} points")
    return best_C, best_R, best_inlier_indices

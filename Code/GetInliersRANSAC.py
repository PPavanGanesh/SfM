import numpy as np
from EstimateFundamentalMatrix import EstimateFundamentalMatrix

def calculate_epipolar_distance(F, pts1, pts2):
    """
    Calculate the epipolar distance for each point correspondence.
    
    Args:
        F: 3x3 fundamental matrix
        pts1: Nx2 array of points in the first image
        pts2: Nx2 array of corresponding points in the second image
        
    Returns:
        distances: Nx1 array of epipolar distances
    """
    # Convert points to homogeneous coordinates
    pts1_homog = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homog = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    
    # Calculate epipolar lines: l' = F*x
    epipolar_lines = pts1_homog @ F.T
    
    # Calculate distances: d = |x'^T * F * x| / sqrt((F*x)_1^2 + (F*x)_2^2)
    # Where (F*x)_i is the i-th component of F*x
    numerator = np.abs(np.sum(pts2_homog * (pts1_homog @ F.T), axis=1))
    denominator = np.sqrt(epipolar_lines[:, 0]**2 + epipolar_lines[:, 1]**2)
    
    # Avoid division by zero
    denominator[denominator < 1e-10] = 1e-10
    
    distances = numerator / denominator
    
    return distances

def GetInliersRANSAC(pts1, pts2, threshold=2.0, num_iterations=2000):
    """
    Find inliers using RANSAC for fundamental matrix estimation.
    
    Args:
        pts1: Nx2 array of points in the first image
        pts2: Nx2 array of corresponding points in the second image
        threshold: Distance threshold for inlier classification
        num_iterations: Number of RANSAC iterations
        
    Returns:
        F: 3x3 fundamental matrix with the most inliers
        inliers: Boolean array indicating inliers
    """
    num_points = pts1.shape[0]
    
    if num_points < 8:
        return None, np.array([])
    
    best_F = None
    best_inliers = np.array([])
    max_inliers = 0
    
    for _ in range(num_iterations):
        # Randomly select 8 points
        indices = np.random.choice(num_points, 8, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]
        
        # Estimate fundamental matrix
        F = EstimateFundamentalMatrix(sample_pts1, sample_pts2)
        
        if F is None:
            continue
        
        # Calculate distances
        distances = calculate_epipolar_distance(F, pts1, pts2)
        
        # Find inliers
        inliers = distances < threshold
        num_inliers = np.sum(inliers)
        
        # Update best model if we found more inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_F = F
            best_inliers = inliers
    
    # Re-estimate F using all inliers
    if max_inliers > 8:
        best_F = EstimateFundamentalMatrix(pts1[best_inliers], pts2[best_inliers])
    
    return best_F, best_inliers

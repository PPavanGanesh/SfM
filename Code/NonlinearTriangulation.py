# import numpy as np
# from scipy.optimize import least_squares

# def NonlinearTriangulation(K, C1, R1, C2, R2, pts1, pts2, X_init):
#     """
#     Refine the 3D points using nonlinear optimization to minimize reprojection error.
    
#     Args:
#         K: 3x3 camera intrinsic matrix
#         C1: 3x1 camera center of the first camera
#         # R1: 3x3 rotation matrix of the first camera
#         C2: 3x1 camera center of the second camera
#         R2: 3x3 rotation matrix of the second camera
#         pts1: Nx2 array of points in the first image
#         pts2: Nx2 array of corresponding points in the second image
#         X_init: Nx3 array of initial 3D points from linear triangulation
        
#     Returns:
#         X_refined: Nx3 array of refined 3D points
#     """
#     if np.allclose(C1, C2) and np.allclose(R1, R2):
#         print("Warning: Cannot perform nonlinear triangulation with identical cameras")
#         return X_init

#     # Compute projection matrices
#     P1 = K @ np.hstack((R1, -R1 @ C1.reshape(3, 1)))
#     P2 = K @ np.hstack((R2, -R2 @ C2.reshape(3, 1)))
    
#     # Refine each 3D point individually
#     X_refined = np.zeros_like(X_init)
    
#     for i in range(X_init.shape[0]):
#         # Get the corresponding 2D points
#         x1 = pts1[i]
#         x2 = pts2[i]
        
#         # Initial 3D point
#         X0 = X_init[i]
        
#         # Define the cost function for reprojection error
#         def reprojection_error(X):
#             # Convert to homogeneous coordinates
#             X_homog = np.append(X, 1)
            
#             # Project 3D point to both cameras
#             x1_proj = P1 @ X_homog
#             x1_proj = x1_proj[:2] / x1_proj[2]  # Normalize
            
#             x2_proj = P2 @ X_homog
#             x2_proj = x2_proj[:2] / x2_proj[2]  # Normalize
            
#             # Compute reprojection error
#             error = np.concatenate([
#                 x1_proj - x1,
#                 x2_proj - x2
#             ])
            
#             return error
        
#         # Perform nonlinear optimization using Levenberg-Marquardt algorithm
#         result = least_squares(reprojection_error, X0, method='lm')
        
#         # Store the refined 3D point
#         X_refined[i] = result.x
    
#     return X_refined

import numpy as np
from scipy.optimize import least_squares

def NonlinearTriangulation(K, C1, R1, C2, R2, pts1, pts2, X_init):
    """
    Refine the 3D points using nonlinear optimization to minimize reprojection error.
    
    Args:
        K: 3x3 camera intrinsic matrix
        C1: 3x1 camera center of the first camera
        R1: 3x3 rotation matrix of the first camera
        C2: 3x1 camera center of the second camera
        R2: 3x3 rotation matrix of the second camera
        pts1: Nx2 array of points in the first image
        pts2: Nx2 array of corresponding points in the second image
        X_init: Nx3 array of initial 3D points from linear triangulation
        
    Returns:
        X_refined: Nx3 array of refined 3D points
    """
    # Compute projection matrices
    P1 = K @ np.hstack((R1, -R1 @ C1.reshape(3, 1)))
    P2 = K @ np.hstack((R2, -R2 @ C2.reshape(3, 1)))
    
    # Refine each 3D point individually
    X_refined = np.zeros_like(X_init)
    
    for i in range(X_init.shape[0]):
        # Get the corresponding 2D points
        x1 = pts1[i]
        x2 = pts2[i]
        
        # Initial 3D point
        X0 = X_init[i]
        
        # Define the cost function for reprojection error
        def reprojection_error(X):
            # Convert to homogeneous coordinates
            X_homog = np.append(X, 1)
            
            # Project 3D point to both cameras
            x1_proj = P1 @ X_homog
            x1_proj = x1_proj[:2] / x1_proj[2]  # Normalize
            
            x2_proj = P2 @ X_homog
            x2_proj = x2_proj[:2] / x2_proj[2]  # Normalize
            
            # Compute reprojection error
            error = np.concatenate([
                x1_proj - x1,
                x2_proj - x2
            ])
            
            return error
        
        # Perform nonlinear optimization using Levenberg-Marquardt algorithm
        try:
            result = least_squares(reprojection_error, X0, method='lm')
            X_refined[i] = result.x
        except ValueError:
            # If optimization fails, keep the initial point
            X_refined[i] = X0
    
    return X_refined

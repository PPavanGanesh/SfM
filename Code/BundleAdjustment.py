import numpy as np
import time
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

def bundle_adjustment_sparsity(visibility_matrix):
    """
    Create sparsity structure for the Jacobian matrix
    
    Args:
        visibility_matrix: IxJ binary matrix where I is the number of cameras and J is the number of points
                          visibility_matrix[i,j] = 1 if point j is visible from camera i, 0 otherwise
    
    Returns:
        A: Sparse matrix defining the sparsity structure of the Jacobian
    """
    n_cameras = visibility_matrix.shape[0]
    n_points = visibility_matrix.shape[1]
    
    camera_indices, point_indices = [], []
    for i in range(n_cameras):
        for j in range(n_points):
            if visibility_matrix[i, j]:
                camera_indices.append(i)
                point_indices.append(j)
    
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    n_observations = len(camera_indices)
    
    m = n_observations * 2  # Each observation contributes 2 residuals (x, y)
    n = n_cameras * 6 + n_points * 3  # 6 params for each camera, 3 for each point
    
    A = lil_matrix((m, n), dtype=int)
    
    i = np.arange(n_observations)
    
    # Fill in camera block
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1
    
    # Fill in point block
    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
    
    return A

def reprojection_error(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """
    Compute residuals (reprojection error)
    
    Args:
        params: Parameter vector containing camera poses and 3D points
        n_cameras: Number of cameras
        n_points: Number of 3D points
        camera_indices: Camera indices for each observation
        point_indices: Point indices for each observation
        points_2d: 2D point observations
        K: Camera intrinsic matrix
    
    Returns:
        error: Reprojection error vector
    """
    # Extract camera parameters and 3D points
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    # Project 3D points to 2D
    points_proj = np.zeros((len(camera_indices), 2))
    
    for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):

        if cam_idx >= n_cameras or pt_idx >= n_points:
            # Skip invalid indices
            continue
        # Extract camera parameters
        rot_vec = camera_params[cam_idx, :3]
        C = camera_params[cam_idx, 3:].reshape(3, 1)
        
        # Convert rotation vector to matrix
        R = Rotation.from_rotvec(rot_vec).as_matrix()
        
        # Extract 3D point
        X = points_3d[pt_idx]
        
        # Project point
        P = K @ np.hstack((R, -R @ C))
        X_homog = np.append(X, 1)
        x_proj = P @ X_homog
        x_proj = x_proj[:2] / x_proj[2]
        
        points_proj[i] = x_proj
    
    # Compute error
    error = (points_proj - points_2d).ravel()
    return error

def validate_bundle_adjustment_inputs(camera_poses, points_3d, visibility_matrix):
    n_cameras = visibility_matrix.shape[0]
    n_points = visibility_matrix.shape[1]
    
    # Check if visibility matrix dimensions match available cameras and points
    if n_cameras > len(camera_poses):
        print(f"Warning: Visibility matrix has {n_cameras} cameras but only {len(camera_poses)} are available")
        # Truncate visibility matrix
        visibility_matrix = visibility_matrix[:len(camera_poses), :]
    
    if n_points > points_3d.shape[0]:
        print(f"Warning: Visibility matrix has {n_points} points but only {points_3d.shape[0]} are available")
        # Truncate visibility matrix
        visibility_matrix = visibility_matrix[:, :points_3d.shape[0]]
    
    return visibility_matrix


def BundleAdjustment(camera_poses, points_3d, K, visibility_matrix):
    """
    Refine camera poses and 3D points by minimizing reprojection error.
    
    Args:
        camera_poses: Dictionary of camera poses, each as a tuple (C, R)
        points_3d: Nx3 array of 3D points
        K: 3x3 camera intrinsic matrix
        visibility_matrix: IxJ binary matrix where I is the number of cameras and J is the number of points
    
    Returns:
        refined_camera_poses: Dictionary of refined camera poses
        refined_points_3d: Nx3 array of refined 3D points
    """

    if isinstance(visibility_matrix, tuple):
        visibility_matrix = visibility_matrix[0]

    # Extract camera indices and convert camera poses to parameter vector
    camera_indices = list(camera_poses.keys())
    n_cameras = len(camera_indices)
    n_points = points_3d.shape[0]

    visibility_matrix = validate_bundle_adjustment_inputs(camera_poses, points_3d, visibility_matrix)
    
    # Initialize parameter vector
    camera_params = np.zeros(n_cameras * 6)  # 3 for rotation vector, 3 for translation
    
    # Fill camera parameters
    for i, idx in enumerate(camera_indices):
        C, R = camera_poses[idx]
        # Convert rotation matrix to rotation vector
        rot_vec = Rotation.from_matrix(R).as_rotvec()
        camera_params[i * 6:i * 6 + 3] = rot_vec
        # camera_params[i * 6 + 3:i * 6 + 6] = C
        camera_params[i * 6 + 3:i * 6 + 6] = C.flatten()
    
    # Fill point parameters
    point_params = points_3d.flatten()
    
    # Combine parameters
    params = np.hstack((camera_params, point_params))
    
    # # Get camera and point indices from visibility matrix
    # cam_indices, pt_indices = [], []
    # for i in range(visibility_matrix.shape[0]):
    #     for j in range(visibility_matrix.shape[1]):
    #         if visibility_matrix[i, j]:
    #             cam_indices.append(i)
    #             pt_indices.append(j)
    
    # cam_indices = np.array(cam_indices)
    # pt_indices = np.array(pt_indices)
    ################################# changed here today morning, adding filtering ############################

    # Get camera and point indices from visibility matrix
    cam_indices, pt_indices = [], []
    for i in range(visibility_matrix.shape[0]):
        for j in range(visibility_matrix.shape[1]):
            if visibility_matrix[i, j]:
                # Only include valid indices
                if i < len(camera_indices) and j < points_3d.shape[0]:
                    cam_indices.append(i)
                    pt_indices.append(j)
                else:
                    print(f"Skipping invalid index pair: camera {i}, point {j}")

    cam_indices = np.array(cam_indices)
    pt_indices = np.array(pt_indices)


    # Create 2D points array from visible points
    points_2d = np.zeros((len(cam_indices), 2))
    for i, (cam_idx, pt_idx) in enumerate(zip(cam_indices, pt_indices)):
        if cam_idx < len(camera_indices) and camera_indices[cam_idx] in camera_poses:
            if pt_idx < points_3d.shape[0]:
                # Project 3D point to camera
                C, R = camera_poses[camera_indices[cam_idx]]
                X = points_3d[pt_idx]
                X_homog = np.append(X, 1)
                
                # Compute projection matrix
                P = K @ np.hstack((R, -R @ C.reshape(3, 1)))
                
                # Project point
                x_proj = P @ X_homog
                x_proj = x_proj[:2] / x_proj[2]
                
                points_2d[i] = x_proj
            else:
                print(f"Warning: Point index {pt_idx} out of bounds for points_3d with size {points_3d.shape[0]}")
                points_2d[i] = [0, 0]  # Set to default value
        else:
            print(f"Warning: Invalid camera index {cam_idx} or camera not found in poses")
            # Skip this point or set to default value
            points_2d[i] = [0, 0]  # Set to origin as default

    
    # # Create 2D points array from visible points
    # points_2d = np.zeros((len(cam_indices), 2))
    # for i, (cam_idx, pt_idx) in enumerate(zip(cam_indices, pt_indices)):
    #     if cam_idx < len(camera_indices) and camera_indices[cam_idx] in camera_poses:
    #     # Project 3D point to camera
    #         C, R = camera_poses[camera_indices[cam_idx]]
    #         X = points_3d[pt_idx]
    #         X_homog = np.append(X, 1)
            
    #         # Compute projection matrix
    #         P = K @ np.hstack((R, -R @ C.reshape(3, 1)))
            
    #         # Project point
    #         x_proj = P @ X_homog
    #         x_proj = x_proj[:2] / x_proj[2]
            
    #         points_2d[i] = x_proj
    #     else:
    #         print(f"Warning: Invalid camera index {cam_idx} or camera not found in poses")
    #         # Skip this point or set to default value
    #         points_2d[i] = [0, 0]  # Set to origin as default

    #     # Project 3D point to camera
    #     C, R = camera_poses[camera_indices[cam_idx]]
    #     X = points_3d[pt_idx]
    #     X_homog = np.append(X, 1)
        
    #     # Compute projection matrix
    #     P = K @ np.hstack((R, -R @ C.reshape(3, 1)))
        
    #     # Project point
    #     x_proj = P @ X_homog
    #     x_proj = x_proj[:2] / x_proj[2]
        
    #     points_2d[i] = x_proj
    
    # Create sparsity structure for the Jacobian
    A = bundle_adjustment_sparsity(visibility_matrix)
    
    # Perform optimization
    t0 = time.time()
    result = least_squares(
        reprojection_error, 
        params, 
        jac_sparsity=A, 
        verbose=2, 
        x_scale='jac', 
        ftol=1e-4, 
        method='trf',
        max_nfev = 500,
        args=(n_cameras, n_points, cam_indices, pt_indices, points_2d, K)
    )
    t1 = time.time()
    print(f"Bundle adjustment took {t1-t0:.2f} seconds")
    
    # Extract optimized parameters
    optimized_params = result.x
    camera_params = optimized_params[:n_cameras * 6].reshape(n_cameras, 6)
    point_params = optimized_params[n_cameras * 6:].reshape(n_points, 3)
    
    # Convert back to camera poses and 3D points
    refined_camera_poses = {}
    for i, idx in enumerate(camera_indices):
        rot_vec = camera_params[i, :3]
        C = camera_params[i, 3:6]
        R = Rotation.from_rotvec(rot_vec).as_matrix()
        refined_camera_poses[idx] = (C, R)
    
    refined_points_3d = point_params
    
    return refined_camera_poses, refined_points_3d

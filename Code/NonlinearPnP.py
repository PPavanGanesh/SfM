import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

def NonlinearPnP(X, x, K, C_init, R_init):
    """
    Refine camera pose using nonlinear optimization to minimize reprojection error.
    
    Args:
        X: Nx3 array of 3D points in world coordinates
        x: Nx2 array of 2D points in image coordinates
        K: 3x3 camera intrinsic matrix
        C_init: 3x1 initial camera center from linear PnP
        R_init: 3x3 initial rotation matrix from linear PnP
        
    Returns:
        C: 3x1 refined camera center
        R: 3x3 refined rotation matrix
    """
    # Convert initial rotation matrix to quaternion
    r = R.from_matrix(R_init)
    q_init = r.as_quat()  # [x, y, z, w] format
    
    # Initial parameters (quaternion and camera center)
    params_init = np.concatenate([q_init, C_init.flatten()])
    
    # Define cost function for reprojection error
    def reprojection_error(params):
        # Extract quaternion and camera center
        q = params[:4]
        C = params[4:].reshape(3, 1)
        
        # Convert quaternion to rotation matrix
        r = R.from_quat(q)
        rot_matrix = r.as_matrix()
        
        # Compute projection matrix P = K[R|-RC]
        P = K @ np.hstack((rot_matrix, -rot_matrix @ C))
        
        # Extract rows of P
        P1 = P[0, :]
        P2 = P[1, :]
        P3 = P[2, :]
        
        # Compute reprojection error for all points
        errors = []
        for i in range(len(X)):
            # Convert to homogeneous coordinates
            X_homog = np.append(X[i], 1)
            
            # Project 3D point to image
            u_proj = np.dot(P1, X_homog) / np.dot(P3, X_homog)
            v_proj = np.dot(P2, X_homog) / np.dot(P3, X_homog)
            
            # Compute error
            u_err = x[i, 0] - u_proj
            v_err = x[i, 1] - v_proj
            
            errors.extend([u_err, v_err])
        
        return np.array(errors)
    
    # Perform nonlinear optimization
    result = least_squares(reprojection_error, params_init, method='trf') ##### 'trf' is the Trust Region Reflective algorithm
    
    # Extract optimized parameters
    params_opt = result.x
    q_opt = params_opt[:4]
    C_opt = params_opt[4:].reshape(3, 1)
    
    # Convert quaternion back to rotation matrix
    r_opt = R.from_quat(q_opt)
    R_opt = r_opt.as_matrix()
    
    return C_opt, R_opt

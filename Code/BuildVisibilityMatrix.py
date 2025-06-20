# import numpy as np

# def BuildVisibilityMatrix(feature_flag, X_found=None):
#     """
#     Build a visibility matrix that shows which 3D points are visible in which cameras.
    
#     Args:
#         feature_flag: NxM array where N is the number of features and M is the number of cameras
#                      feature_flag[i,j] = 1 if feature i is visible in camera j, 0 otherwise
#         X_found: Optional binary array indicating which 3D points have been successfully triangulated
        
#     Returns:
#         V: IxJ binary matrix where I is the number of cameras and J is the number of points
#            V[i,j] = 1 if point j is visible from camera i, 0 otherwise
#         X_index: Indices of points that are both found and visible (if X_found is provided)
#     """
#     # Get the number of cameras (I) and features (J)
#     num_features, num_cameras = feature_flag.shape
    
#     # If X_found is provided, filter points
#     if X_found is not None:
#         # Find points visible in at least one camera
#         visible_in_any = np.any(feature_flag, axis=1)
        
#         # Find points that are both found and visible
#         X_index = np.where(X_found & visible_in_any)[0]
        
#         # Create filtered visibility matrix
#         V = feature_flag[X_index].T
        
#         return V, X_index
    
#     # Otherwise, just build the full visibility matrix
#     V = feature_flag.T
    
#     return V

import numpy as np

def getObservationsIndexAndVizMat(X_found, filtered_feature_flag, nCam):
    """
    Find the 3D points that are visible in cameras up to nCam and create a visibility matrix.
    
    Args:
        X_found: Binary array indicating which 3D points have been successfully triangulated
        filtered_feature_flag: NxM array where N is the number of features and M is the number of cameras
                              filtered_feature_flag[i,j] = 1 if feature i is visible in camera j, 0 otherwise
        nCam: Index of the last camera to consider
        
    Returns:
        X_index: Indices of points that are both found and visible in at least one camera
        visibility_matrix: Binary matrix where rows are points and columns are cameras
                          visibility_matrix[i,j] = 1 if point i is visible from camera j, 0 otherwise
    """
    # Find the 3D points visible in any camera up to nCam
    bin_temp = np.zeros((filtered_feature_flag.shape[0]), dtype=int)
    for n in range(nCam + 1):
        bin_temp = bin_temp | filtered_feature_flag[:,n]

    # Find points that are both triangulated and visible
    X_index = np.where((X_found.reshape(-1)) & (bin_temp))
    
    # Create visibility matrix
    visibility_matrix = X_found[X_index].reshape(-1,1)
    for n in range(nCam + 1):
        visibility_matrix = np.hstack((visibility_matrix, filtered_feature_flag[X_index, n].reshape(-1,1)))

    # Remove the first column (X_found) to get just the visibility information
    o, c = visibility_matrix.shape
    return X_index, visibility_matrix[:, 1:c]

# def BuildVisibilityMatrix(feature_flag, X_found=None):
#     """
#     Build a visibility matrix that shows which 3D points are visible in which cameras.
    
#     Args:
#         feature_flag: NxM array where N is the number of features and M is the number of cameras
#                      feature_flag[i,j] = 1 if feature i is visible in camera j, 0 otherwise
#         X_found: Optional binary array indicating which 3D points have been successfully triangulated
        
#     Returns:
#         V: IxJ binary matrix where I is the number of cameras and J is the number of points
#            V[i,j] = 1 if point j is visible from camera i, 0 otherwise
#         X_index: Indices of points that are both found and visible (if X_found is provided)
#     """
#     # Get the number of cameras (I) and features (J)
#     num_features, num_cameras = feature_flag.shape
    
#     # If X_found is provided, filter points
#     if X_found is not None:
#         # Find points visible in at least one camera
#         visible_in_any = np.any(feature_flag, axis=1)
        
#         # Find points that are both found and visible
#         X_index = np.where(X_found & visible_in_any)[0]
        
#         # Create filtered visibility matrix
#         V = feature_flag[X_index].T
        
#         return V, X_index
    
#     # Otherwise, just build the full visibility matrix
#     V = feature_flag.T
    
    # return V

######################################################## update the function ########################################################

def BuildVisibilityMatrix(feature_flag, X_found=None):
    # Get the number of cameras (I) and features (J)
    num_features, num_cameras = feature_flag.shape
    
    # If X_found is provided, filter points
    if X_found is not None:
        # Find points visible in at least one camera
        visible_in_any = np.any(feature_flag, axis=1)
        
        # Find points that are both found and visible
        X_index = np.where(X_found & visible_in_any)[0]
        
        # Create a mapping from original feature indices to condensed indices
        index_mapping = np.full(num_features, -1)
        for i, idx in enumerate(X_index):
            index_mapping[idx] = i
        
        # Create filtered visibility matrix with proper indices
        V = np.zeros((num_cameras, len(X_index)), dtype=int)
        for i in range(num_cameras):
            for j, idx in enumerate(X_index):
                if feature_flag[idx, i]:
                    V[i, j] = 1
        
        return V, X_index
    
    # Otherwise, just build the full visibility matrix
    V = feature_flag.T
    
    return V

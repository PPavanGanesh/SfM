import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from mpl_toolkits.mplot3d import Axes3D
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from GetInliersRANSAC import GetInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from NonlinearTriangulation import NonlinearTriangulation
from LinearPnP import LinearPnP
from PnPRANSAC import PnPRANSAC
from NonlinearPnP import NonlinearPnP
from BuildVisibilityMatrix import BuildVisibilityMatrix
from BundleAdjustment import BundleAdjustment, validate_bundle_adjustment_inputs

def features_extraction(data_path):
    """
    Extract features from matching files
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        feature_x: x-coordinates of features
        feature_y: y-coordinates of features
        feature_flag: Binary flags indicating feature visibility
        feature_rgb_values: RGB values of features
    """
    no_of_images = 5
    feature_rgb_values = []
    feature_x = []
    feature_y = []
    feature_flag = []
    
    for n in range(1, no_of_images):
        file = os.path.join(data_path, f"matching{n}.txt")
        matching_file = open(file, "r")
        nfeatures = 0
        
        for i, row in enumerate(matching_file):
            if i == 0:  # 1st row having nFeatures
                row_elements = row.split(':')
                nfeatures = int(row_elements[1])
            else:
                # Create arrays to store coordinates and flags
                x_row = np.zeros((1, no_of_images))
                y_row = np.zeros((1, no_of_images))
                flag_row = np.zeros((1, no_of_images), dtype=int)
                
                # Parse the row data
                row_elements = row.split()
                columns = [float(x) for x in row_elements]
                columns = np.asarray(columns)
                
                # Extract feature information
                nMatches = int(columns[0])
                r_value, g_value, b_value = columns[1:4]
                feature_rgb_values.append([r_value, g_value, b_value])
                
                # Store current image coordinates
                current_x, current_y = columns[4:6]
                x_row[0, n-1] = current_x
                y_row[0, n-1] = current_y
                flag_row[0, n-1] = 1
                
                # Process matches in other images
                m = 1
                while m < 3*nMatches and m+7 <= len(columns):
                    image_id = int(columns[5+m])
                    image_id_x = columns[6+m]
                    image_id_y = columns[7+m]
                    m = m+3
                    
                    x_row[0, image_id - 1] = image_id_x
                    y_row[0, image_id - 1] = image_id_y
                    flag_row[0, image_id - 1] = 1
                
                feature_x.append(x_row)
                feature_y.append(y_row)
                feature_flag.append(flag_row)
    
    feature_x = np.asarray(feature_x).reshape(-1, no_of_images)
    feature_y = np.asarray(feature_y).reshape(-1, no_of_images)
    feature_flag = np.asarray(feature_flag).reshape(-1, no_of_images)
    feature_rgb_values = np.asarray(feature_rgb_values).reshape(-1, 3)
    
    return feature_x, feature_y, feature_flag, feature_rgb_values


def draw_matches(img1, img2, pts1, pts2, inliers=None):
    """
    Draw matches between two images
    
    Args:
        img1, img2: Input images
        pts1, pts2: Corresponding points
        inliers: Boolean array indicating inliers
        
    Returns:
        match_img: Image with matches drawn
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    match_img = np.zeros((h, w, 3), dtype=np.uint8)
    match_img[:h1, :w1] = img1
    match_img[:h2, w1:w1+w2] = img2
    
    if inliers is None:
        inliers = np.ones(pts1.shape[0], dtype=bool)
    
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        pt2_adj = (int(pt2[0]) + w1, int(pt2[1]))
        color = (0, 255, 0) if inliers[i] else (0, 0, 255)  # Green for inliers, red for outliers
        
        cv2.line(match_img, (int(pt1[0]), int(pt1[1])), pt2_adj, color, 1)
        cv2.circle(match_img, (int(pt1[0]), int(pt1[1])), 3, color, -1)
        cv2.circle(match_img, pt2_adj, 3, color, -1)
    
    return match_img

def draw_epipolar_lines(img1, img2, pts1, pts2, F, inliers=None):
    """
    Draw epipolar lines on images
    
    Args:
        img1, img2: Input images
        pts1, pts2: Corresponding points
        F: Fundamental matrix
        inliers: Boolean array indicating inliers
        
    Returns:
        fig: Matplotlib figure
        img1_lines, img2_lines: Images with epipolar lines
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    img1_lines = img1.copy()
    img2_lines = img2.copy()
    
    if inliers is None:
        inliers = np.ones(pts1.shape[0], dtype=bool)
    
    pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_hom = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        if inliers[i]:
            color = (0, 255, 0)  # Green for inliers
        else:
            color = (0, 0, 255)  # Red for outliers
            
        cv2.circle(img1_lines, (int(pt1[0]), int(pt1[1])), 3, color, -1)
        cv2.circle(img2_lines, (int(pt2[0]), int(pt2[1])), 3, color, -1)
        
        if inliers[i]:
            # Epipolar line in second image
            line2 = F @ pts1_hom[i]
            a2, b2, c2 = line2
            
            if abs(b2) > 1e-8:
                x_start, x_end = 0, w2
                y_start = int((-a2 * x_start - c2) / b2)
                y_end = int((-a2 * x_end - c2) / b2)
                
                if 0 <= y_start < h2 and 0 <= y_end < h2:
                    cv2.line(img2_lines, (x_start, y_start), (x_end, y_end), color, 1)
            
            # Epipolar line in first image
            line1 = F.T @ pts2_hom[i]
            a1, b1, c1 = line1
            
            if abs(b1) > 1e-8:
                x_start, x_end = 0, w1
                y_start = int((-a1 * x_start - c1) / b1)
                y_end = int((-a1 * x_end - c1) / b1)
                
                if 0 <= y_start < h1 and 0 <= y_end < h1:
                    cv2.line(img1_lines, (x_start, y_start), (x_end, y_end), color, 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(cv2.cvtColor(img1_lines, cv2.COLOR_BGR2RGB))
    ax1.set_title('Image 1 with Epipolar Lines')
    ax2.imshow(cv2.cvtColor(img2_lines, cv2.COLOR_BGR2RGB))
    ax2.set_title('Image 2 with Epipolar Lines')
    plt.tight_layout()
    
    return fig, img1_lines, img2_lines

def visualize_triangulation_comparison(images, pts1, pts2, K, C1, R1, C2, R2, X_linear, X_nonlinear, output_dir, img_pair):
    """
    Visualize comparison between linear and non-linear triangulation
    
    Args:
        images: List of two images
        pts1, pts2: Corresponding points
        K: Camera calibration matrix
        C1, R1: Camera 1 pose
        C2, R2: Camera 2 pose
        X_linear: 3D points from linear triangulation
        X_nonlinear: 3D points from non-linear triangulation
        output_dir: Output directory
        img_pair: String identifier for the image pair
    """
    # Project 3D points back to images
    P1 = K @ np.hstack((R1, -R1 @ C1.reshape(3, 1)))
    P2 = K @ np.hstack((R2, -R2 @ C2.reshape(3, 1)))
    
    # Linear triangulation projections
    pts1_linear_proj = []
    pts2_linear_proj = []
    
    for X in X_linear:
        X_homog = np.append(X, 1)
        x1_proj = P1 @ X_homog
        x1_proj = x1_proj / x1_proj[2]
        pts1_linear_proj.append(x1_proj[:2])
        
        x2_proj = P2 @ X_homog
        x2_proj = x2_proj / x2_proj[2]
        pts2_linear_proj.append(x2_proj[:2])
    
    pts1_linear_proj = np.array(pts1_linear_proj)
    pts2_linear_proj = np.array(pts2_linear_proj)
    
    # Non-linear triangulation projections
    pts1_nonlinear_proj = []
    pts2_nonlinear_proj = []
    
    for X in X_nonlinear:
        X_homog = np.append(X, 1)
        x1_proj = P1 @ X_homog
        x1_proj = x1_proj / x1_proj[2]
        pts1_nonlinear_proj.append(x1_proj[:2])
        
        x2_proj = P2 @ X_homog
        x2_proj = x2_proj / x2_proj[2]
        pts2_nonlinear_proj.append(x2_proj[:2])
    
    pts1_nonlinear_proj = np.array(pts1_nonlinear_proj)
    pts2_nonlinear_proj = np.array(pts2_nonlinear_proj)
    
    # Calculate reprojection errors
    error_linear_img1 = np.mean(np.sqrt(np.sum((pts1 - pts1_linear_proj)**2, axis=1)))
    error_linear_img2 = np.mean(np.sqrt(np.sum((pts2 - pts2_linear_proj)**2, axis=1)))
    
    error_nonlinear_img1 = np.mean(np.sqrt(np.sum((pts1 - pts1_nonlinear_proj)**2, axis=1)))
    error_nonlinear_img2 = np.mean(np.sqrt(np.sum((pts2 - pts2_nonlinear_proj)**2, axis=1)))
    
    print(f"Linear triangulation - Mean reprojection error: Image 1: {error_linear_img1:.2f}px, Image 2: {error_linear_img2:.2f}px")
    print(f"Non-linear triangulation - Mean reprojection error: Image 1: {error_nonlinear_img1:.2f}px, Image 2: {error_nonlinear_img2:.2f}px")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Linear triangulation - Image 1
    axes[0, 0].imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
    axes[0, 0].scatter(pts1[:, 0], pts1[:, 1], c='g', s=5, label='Original points')
    axes[0, 0].scatter(pts1_linear_proj[:, 0], pts1_linear_proj[:, 1], c='r', s=5, label='Reprojected points')
    axes[0, 0].set_title(f'Image 1 - Linear Triangulation (Error: {error_linear_img1:.2f}px)')
    axes[0, 0].legend()
    
    # Linear triangulation - Image 2
    axes[0, 1].imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))
    axes[0, 1].scatter(pts2[:, 0], pts2[:, 1], c='g', s=5, label='Original points')
    axes[0, 1].scatter(pts2_linear_proj[:, 0], pts2_linear_proj[:, 1], c='r', s=5, label='Reprojected points')
    axes[0, 1].set_title(f'Image 2 - Linear Triangulation (Error: {error_linear_img2:.2f}px)')
    axes[0, 1].legend()
    
    # Non-linear triangulation - Image 1
    axes[1, 0].imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
    axes[1, 0].scatter(pts1[:, 0], pts1[:, 1], c='g', s=5, label='Original points')
    axes[1, 0].scatter(pts1_nonlinear_proj[:, 0], pts1_nonlinear_proj[:, 1], c='r', s=5, label='Reprojected points')
    axes[1, 0].set_title(f'Image 1 - Non-linear Triangulation (Error: {error_nonlinear_img1:.2f}px)')
    axes[1, 0].legend()
    
    # Non-linear triangulation - Image 2
    axes[1, 1].imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))
    axes[1, 1].scatter(pts2[:, 0], pts2[:, 1], c='g', s=5, label='Original points')
    axes[1, 1].scatter(pts2_nonlinear_proj[:, 0], pts2_nonlinear_proj[:, 1], c='r', s=5, label='Reprojected points')
    axes[1, 1].set_title(f'Image 2 - Non-linear Triangulation (Error: {error_nonlinear_img2:.2f}px)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'reprojection_comparison_{img_pair}.png'))
    plt.close()

def visualize_final_reconstruction(camera_poses, points_3d, points_3d_before_BA, output_dir):
    """
    Visualize the final reconstruction with all cameras and 3D points
    
    Args:
        camera_poses: Dictionary of camera poses, each as a tuple (C, R)
        points_3d: Nx3 array of 3D points after bundle adjustment
        points_3d_before_BA: Nx3 array of 3D points before bundle adjustment
        output_dir: Output directory
    """
    # Create figure for 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_max, x_min = 10,-10
    z_max, z_min = 20,-20
    
    # Plot 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=1, alpha=0.5, label='After BA')
    ax.scatter(points_3d_before_BA[:, 0], points_3d_before_BA[:, 1], points_3d_before_BA[:, 2], c='red', s=1, alpha=0.5, label='Before BA')
    
    # Plot camera poses
    for cam_idx, (C, R) in camera_poses.items():
        # Plot camera center
        ax.scatter(C[0], C[1], C[2], c='red', marker='o', s=100, label=f'Camera {cam_idx}' if cam_idx == 1 else "")
        
        # Plot camera axes
        scale = 0.5
        ax.quiver(C[0], C[1], C[2], R[0,0]*scale, R[1,0]*scale, R[2,0]*scale, color='r')
        ax.quiver(C[0], C[1], C[2], R[0,1]*scale, R[1,1]*scale, R[2,1]*scale, color='g')
        ax.quiver(C[0], C[1], C[2], R[0,2]*scale, R[1,2]*scale, R[2,2]*scale, color='b')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Reconstruction with Camera Poses')
    
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'reconstruction_3d.png'))
    plt.close()
    
    # Create top-down view (X-Z plane)
    plt.figure(figsize=(12, 10))
    
    # Plot 3D points from top view
    plt.scatter(points_3d[:, 0], points_3d[:, 2], c='blue', s=3, alpha=0.5, label='After BA')
    plt.scatter(points_3d_before_BA[:, 0], points_3d_before_BA[:, 2], c='red', s=1, alpha=1, label='Before BA')
    
    # Plot camera positions from top view
    for cam_idx, (C, R) in camera_poses.items():
        # Create a triangle for each camera
        triangle_height = 0.8 
        triangle_width = 0.2
        
        # Define the triangle vertices
        triangle_x = [C[0], C[0] - triangle_width/2, C[0] + triangle_width/2]
        triangle_z = [C[2] + triangle_height, C[2], C[2]]
        
        # Plot the triangle with a distinct color for each camera
        color = plt.cm.tab10(cam_idx % 10)  # Use a colormap for distinct colors
        plt.fill(triangle_x, triangle_z, color=color, alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Add camera label
        # plt.text(C[0], C[2] - 0.2, f'Camera {cam_idx}', fontsize=10, ha='center')
    
    plt.xlim(x_min, x_max)
    plt.ylim(z_min, z_max)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Top-down View of Reconstruction')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'reconstruction_top_view.png'))
    plt.close()
    
    # Create side view (Y-Z plane)
    plt.figure(figsize=(12, 10))
    
    # Plot 3D points from side view
    plt.scatter(points_3d[:, 1], points_3d[:, 2], c='blue', s=2, alpha=0.5, label='After BA')
    plt.scatter(points_3d_before_BA[:, 1], points_3d_before_BA[:, 2], c='red', s=1, alpha=0.9, label='Before BA')
    
    # Plot camera positions from side view
    for cam_idx, (C, R) in camera_poses.items():
        plt.scatter(C[1], C[2], c='red', marker='^', s=100)
        plt.text(C[1], C[2] - 0.2, f'Camera {cam_idx}', fontsize=10, ha='center')
    
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('Side View of Reconstruction')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'reconstruction_side_view.png'))
    plt.close()

# def visualize_pnp_results(image, X, x, K, C, R, inliers):
#     # Create a copy of the image for visualization
#     vis_img = image.copy()
    
#     # Project all 3D points using the estimated pose
#     P = K @ np.hstack((R, -R @ C.reshape(3, 1)))
    
#     # Draw all points
#     for i in range(len(X)):
#         # Convert 3D point to homogeneous coordinates
#         X_homog = np.append(X[i], 1)
        
#         # Project 3D point to image
#         x_proj = P @ X_homog
#         x_proj = x_proj[:2] / x_proj[2]
        
#         # Draw original point in red
#         cv2.circle(vis_img, (int(x[i, 0]), int(x[i, 1])), 3, (0, 0, 255), -1)
        
#         # Draw projected point in blue
#         cv2.circle(vis_img, (int(x_proj[0]), int(x_proj[1])), 3, (255, 0, 0), -1)
        
#         # Draw line between original and projected point
#         cv2.line(vis_img, (int(x[i, 0]), int(x[i, 1])), 
#                  (int(x_proj[0]), int(x_proj[1])), (0, 255, 0), 1)
        
#         # Highlight inliers with green circles
#         if i in inliers:
#             cv2.circle(vis_img, (int(x[i, 0]), int(x[i, 1])), 5, (0, 255, 0), 1)
    
#     # Draw camera pose (coordinate axes)
#     origin = np.array([0, 0, 0, 1])
#     x_axis = np.array([1, 0, 0, 1])
#     y_axis = np.array([0, 1, 0, 1])
#     z_axis = np.array([0, 0, 1, 1])
    
#     o_proj = P @ origin
#     o_proj = o_proj[:2] / o_proj[2]
    
#     x_proj = P @ x_axis
#     x_proj = x_proj[:2] / x_proj[2]
    
#     y_proj = P @ y_axis
#     y_proj = y_proj[:2] / y_proj[2]
    
#     z_proj = P @ z_axis
#     z_proj = z_proj[:2] / z_proj[2]
    
#     # Draw coordinate axes
#     cv2.line(vis_img, (int(o_proj[0]), int(o_proj[1])), 
#              (int(x_proj[0]), int(x_proj[1])), (0, 0, 255), 2)  # X-axis in red
#     cv2.line(vis_img, (int(o_proj[0]), int(o_proj[1])), 
#              (int(y_proj[0]), int(y_proj[1])), (0, 255, 0), 2)  # Y-axis in green
#     cv2.line(vis_img, (int(o_proj[0]), int(o_proj[1])), 
#              (int(z_proj[0]), int(z_proj[1])), (255, 0, 0), 2)  # Z-axis in blue
    
#     return vis_img

# def visualize_camera_poses(camera_poses, points_3d, output_dir, filename="camera_poses.png"):
#     """
#     Visualize camera poses and 3D points in the X-Z plane (top-down view)
    
#     Args:
#         camera_poses: Dictionary of camera poses, each as a tuple (C, R)
#         points_3d: Nx3 array of 3D points
#         output_dir: Output directory
#         filename: Output filename
#     """
#     plt.figure(figsize=(10, 8))
    
#     # Plot 3D points from top view (X-Z plane)
#     plt.scatter(points_3d[:, 0], points_3d[:, 2], c='blue', s=1, alpha=0.5)
    
#     # Plot camera positions as triangles
#     camera_colors = {
#         1: 'black',
#         2: 'blue',
#         3: 'orange',
#         4: 'green',
#         5: 'red'
#     }
    
#     # Add legend entries
#     legend_elements = []
    
#     for cam_idx, (C, R) in camera_poses.items():
#         # Create a triangle for each camera
#         triangle_height = 0.5
#         triangle_width = 0.3
        
#         # Define the triangle vertices
#         triangle_x = [C[0], C[0] - triangle_width/2, C[0] + triangle_width/2]
#         triangle_z = [C[2] + triangle_height, C[2], C[2]]
        
#         # Plot the triangle with a distinct color for each camera
#         color = camera_colors.get(cam_idx, 'gray')
#         plt.fill(triangle_x, triangle_z, color=color, alpha=0.7, edgecolor='black', linewidth=1)
        
#         # Add camera label
#         plt.text(C[0], C[2] - 0.2, f'Camera {cam_idx}', fontsize=10, ha='center')
        
#         # Add to legend
#         legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
#                                          markerfacecolor=color, markersize=10,
#                                          label=f'Camera {cam_idx}'))
    
#     # Set axis limits to focus on the scene
#     x_min, x_max = min(points_3d[:, 0]), max(points_3d[:, 0])
#     z_min, z_max = min(points_3d[:, 2]), max(points_3d[:, 2])
    
#     # Add some padding
#     x_range = x_max - x_min
#     z_range = z_max - z_min
#     plt.xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
#     plt.ylim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)
    
#     plt.xlabel('X')
#     plt.ylabel('Z')
#     plt.title(f'SfM with Views {", ".join(str(k) for k in sorted(camera_poses.keys()))}')
#     plt.legend(handles=legend_elements, loc='upper left')
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.savefig(os.path.join(output_dir, filename), dpi=300)
#     plt.close()
###############################################################################################################3

def visualize_camera_poses(camera_poses, points_3d, output_dir, filename="camera_poses.png"):
    """
    Visualize camera poses and 3D points in the X-Z plane (top-down view)
    """
    plt.figure(figsize=(10, 8))
    
    # Filter outlier points
    filtered_points = filter_outlier_points(points_3d)
    
    # Plot 3D points from top view (X-Z plane)
    plt.scatter(filtered_points[:, 0], filtered_points[:, 2], c='blue', s=1, alpha=0.5)
    
    # Plot camera positions as triangles
    camera_colors = {
        1: 'black',
        2: 'blue',
        3: 'orange',
        4: 'green',
        5: 'red'
    }
    
    # Add legend entries
    legend_elements = []
    
    for cam_idx, (C, R) in camera_poses.items():
        # Create a triangle for each camera
        triangle_height = 0.8
        triangle_width = 0.2
        
        # Define the triangle vertices
        triangle_x = [C[0], C[0] - triangle_width/2, C[0] + triangle_width/2]
        triangle_z = [C[2] + triangle_height, C[2], C[2]]
        
        # Plot the triangle with a distinct color for each camera
        color = camera_colors.get(cam_idx, 'gray')
        plt.fill(triangle_x, triangle_z, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add camera label
        # plt.text(C[0], C[2] - 0.2, f'Camera {cam_idx}', fontsize=10, ha='center')
        
        # Add to legend
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor=color, markersize=10,
                                        label=f'Camera {cam_idx}'))
    
    # Set fixed axis limits to focus on the cameras
    plt.xlim(-10, 10)
    plt.ylim(-20, 20)
    
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title(f'SfM with Views {", ".join(str(k) for k in sorted(camera_poses.keys()))}')
    plt.legend(handles=legend_elements, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def visualize_initial_triangulation(poses, X_set, output_dir):
    plt.figure(figsize=(8, 8))
    colors = ['red', 'blue', 'green', 'magenta']
    
    for i, (X, (C, R)) in enumerate(zip(X_set, poses)):
        # Plot 3D points
        plt.scatter(X[:, 0], X[:, 2], c=colors[i], s=1, alpha=0.5)
        
        # Plot camera center
        plt.scatter(C[0], C[2], c=colors[i], marker='^', s=100)
    
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Initial triangulation')
    plt.savefig(os.path.join(output_dir, 'initial_triangulation.png'))
    plt.close()

def visualize_disambiguated_pose(poses, X_set, best_pose_idx, output_dir):
    """
    Visualize the disambiguated camera pose and the corresponding 3D points
    
    Args:
        poses: List of four possible camera poses, each as a tuple (C, R)
        X_set: List of four sets of triangulated 3D points
        best_pose_idx: Index of the best pose
        output_dir: Output directory
    """
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'magenta']

    x_min, x_max = -20, 20
    z_min, z_max = -20, 20
    
    # Plot all four possible camera poses and their points
    for i, (X, (C, R)) in enumerate(zip(X_set, poses)):
        # Plot 3D points with lower alpha for non-selected poses
        alpha = 0.8 if i == best_pose_idx else 0.2
        marker_size = 5 if i == best_pose_idx else 2
        
        plt.scatter(X[:, 0], X[:, 2], c=colors[i], s=marker_size, alpha=alpha, 
                   label=f"Pose {i+1}{' (Best)' if i == best_pose_idx else ''}")
        
        # Plot camera center
        plt.scatter(C[0], C[2], c=colors[i], marker='^', s=100, alpha=alpha)
    
    plt.xlim(x_min, x_max)
    plt.ylim(z_min, z_max)
    
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Disambiguated Camera Pose')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'disambiguated_pose.png'))
    plt.close()

# def visualize_disambiguated_pose(poses, X_set, best_pose_idx, output_dir):
#     """
#     Visualize the disambiguated camera pose and the corresponding 3D points
    
#     Args:
#         poses: List of four possible camera poses, each as a tuple (C, R)
#         X_set: List of four sets of triangulated 3D points
#         best_pose_idx: Index of the best pose
#         output_dir: Output directory
#     """
#     plt.figure(figsize=(10, 8))
#     colors = ['red', 'blue', 'green', 'magenta']
    
#     # Define fixed axis limits
#     x_min, x_max = -2, 2
#     z_min, z_max = -2, 2
    
#     # Plot all four possible camera poses and their points
#     for i, (X, (C, R)) in enumerate(zip(X_set, poses)):
#         # Plot 3D points with lower alpha for non-selected poses
#         alpha = 0.8 if i == best_pose_idx else 0.2
#         marker_size = 5 if i == best_pose_idx else 2
        
#         # Filter out extreme outliers to avoid them affecting the visualization
#         valid_indices = np.where((X[:, 0] > x_min*2) & (X[:, 0] < x_max*2) & 
#                                 (X[:, 2] > z_min*2) & (X[:, 2] < z_max*2))[0]
        
#         if len(valid_indices) > 0:
#             plt.scatter(X[valid_indices, 0], X[valid_indices, 2], c=colors[i], s=marker_size, 
#                       alpha=alpha, label=f"Pose {i+1}{' (Best)' if i == best_pose_idx else ''}")
        
#         # Always plot camera center
#         plt.scatter(C[0], C[2], c=colors[i], marker='^', s=100, alpha=alpha)
    
    # Set fixed axis limits
    # plt.xlim(x_min, x_max)
    # plt.ylim(z_min, z_max)
    
    # plt.xlabel('X')
    # plt.ylabel('Z')
    # plt.title('Disambiguated Camera Pose')
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.savefig(os.path.join(output_dir, 'disambiguated_pose.png'))
    # plt.close()


def visualize_linear_nonlinear_comparison(X_linear, X_nonlinear, C1, R1, C2, R2, output_dir):
    """
    Visualize comparison between linear and non-linear triangulation
    
    Args:
        X_linear: 3D points from linear triangulation
        X_nonlinear: 3D points from non-linear triangulation
        C1, R1: Camera 1 pose
        C2, R2: Camera 2 pose
        output_dir: Output directory
    """
    plt.figure(figsize=(10, 8))
    
    # Plot linear triangulation points
    plt.scatter(X_linear[:, 0], X_linear[:, 2], c='red', s=4, alpha=0.9, label='Linear')
    
    # Plot non-linear triangulation points
    plt.scatter(X_nonlinear[:, 0], X_nonlinear[:, 2], c='blue', s=2, alpha=0.5, label='Non-linear')
    
    # Plot camera centers
    plt.scatter(C1[0], C1[2], c='green', marker='^', s=100, label='Camera 1')
    plt.scatter(C2[0], C2[2], c='blue', marker='^', s=100, label='Camera 2')
    
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Linear vs Non-linear Triangulation')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.savefig(os.path.join(output_dir, 'linear_nonlinear_comparison.png'))
    plt.close()

def filter_outlier_points(points_3d, percentile=99):
    """Filter extreme outlier points that distort the visualization"""
    x_vals = points_3d[:, 0]
    y_vals = points_3d[:, 1]
    z_vals = points_3d[:, 2]
    
    # Calculate percentile thresholds
    x_min, x_max = np.percentile(x_vals, [100-percentile, percentile])
    y_min, y_max = np.percentile(y_vals, [100-percentile, percentile])
    z_min, z_max = np.percentile(z_vals, [100-percentile, percentile])
    
    # Filter points within reasonable bounds
    mask = (x_vals > x_min) & (x_vals < x_max) & \
           (y_vals > y_min) & (y_vals < y_max) & \
           (z_vals > z_min) & (z_vals < z_max)
    
    return points_3d[mask]

def calculate_reprojection_error(X_3d, x_2d, K, C, R):
    """
    Calculate the reprojection error for a set of 3D points and their 2D projections
    
    Args:
        X_3d: Nx3 array of 3D points
        x_2d: Nx2 array of 2D points
        K: 3x3 camera intrinsic matrix
        C: Camera center
        R: Rotation matrix
        
    Returns:
        mean_error: Mean reprojection error in pixels
    """
    # Compute projection matrix
    P = K @ np.hstack((R, -R @ C.reshape(3, 1)))
    
    # Project 3D points to 2D
    x_proj = np.zeros_like(x_2d)
    for i in range(X_3d.shape[0]):
        X_homog = np.append(X_3d[i], 1)
        x = P @ X_homog
        x = x / x[2]
        x_proj[i] = x[:2]
    
    # Calculate reprojection error
    errors = np.sqrt(np.sum((x_2d - x_proj)**2, axis=1))
    mean_error = np.mean(errors)
    
    return mean_error


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Outputs', default='./Outputs/', help='Outputs are saved here')
    Parser.add_argument('--Data', default='../P2Data', help='Data')

    Args = Parser.parse_args()
    Data = Args.Data
    Output = Args.Outputs
    
    # Create output directory if it doesn't exist
    os.makedirs(Output, exist_ok=True)

    # Load images
    images = []
    for i in range(1, 6):  # 5 images given
        path = os.path.join(Data, f"{i}.png")
        image = cv2.imread(path)
        if image is not None:
            images.append(image)
        else:
            print(f"No image found at {path}")
            return

    # Read camera calibration matrix
    K = np.loadtxt(os.path.join(Data, 'calibration.txt'))
    print("Camera calibration matrix K:")
    print(K)

    # Extract features from matching files
    feature_x, feature_y, feature_flag, feature_rgb_values = features_extraction(Data)
    
    # Initialize filtered feature flag and matrices to store F and E
    filtered_feature_flag = np.zeros_like(feature_flag)
    f_matrix = np.empty(shape=(5, 5), dtype=object)
    e_matrix = np.empty(shape=(5, 5), dtype=object)
    
    # Dictionary to store camera poses and 3D points
    camera_poses = {}
    points_3d = {}
    
    # Process all image pairs to reject outlier correspondences
    for i in range(0, 4):  # No of Images = 5
        for j in range(i+1, 5):
            if i == j:
                continue
                
            idx = np.where(feature_flag[:, i] & feature_flag[:, j])
            pts1 = np.hstack((feature_x[idx, i].reshape((-1, 1)), feature_y[idx, i].reshape((-1, 1))))
            pts2 = np.hstack((feature_x[idx, j].reshape((-1, 1)), feature_y[idx, j].reshape((-1, 1))))
            idx = np.array(idx).reshape(-1)
            
            if len(idx) > 8:
                # For "before RANSAC" (all green)
                before_ransac_img = draw_matches(images[i], images[j], pts1, pts2)
                cv2.imwrite(os.path.join(Output, f'matches_before_ransac_{i+1}_{j+1}.png'), before_ransac_img)
                
                # Get inliers using RANSAC
                F_ransac, inliers = GetInliersRANSAC(pts1, pts2)
                print(f"Between Images: {i+1} and {j+1}, NO of Inliers: {np.sum(inliers)}/{len(pts1)}")
                
                # For "after RANSAC" (green inliers, red outliers)
                after_ransac_img = draw_matches(images[i], images[j], pts1, pts2, inliers)
                cv2.imwrite(os.path.join(Output, f'matches_after_ransac_{i+1}_{j+1}.png'), after_ransac_img)
                
                # Store the fundamental matrix
                f_matrix[i, j] = F_ransac
                
                # Update filtered feature flag
                for k in range(len(idx)):
                    if inliers[k]:
                        filtered_feature_flag[idx[k], j] = 1
                        filtered_feature_flag[idx[k], i] = 1
                
                # Visualize epipolar lines
                fig, img1_lines, img2_lines = draw_epipolar_lines(images[i], images[j], pts1, pts2, F_ransac, inliers)
                plt.savefig(os.path.join(Output, f'epipolar_lines_{i+1}_{j+1}.png'))
                plt.close(fig)
                
                # Save individual images with epipolar lines
                cv2.imwrite(os.path.join(Output, f'epipolar_img{i+1}_{j+1}.png'), img1_lines)
                cv2.imwrite(os.path.join(Output, f'epipolar_img{j+1}_{i+1}.png'), img2_lines)
                
                # Get only inlier points for further processing
                inlier_pts1 = pts1[inliers]
                inlier_pts2 = pts2[inliers]
                
                # Compute essential matrix from fundamental matrix
                E = EssentialMatrixFromFundamentalMatrix(F_ransac, K)
                e_matrix[i, j] = E
                print(f"Essential Matrix between images {i+1} and {j+1}:")
                print(E)
                
                # Extract camera poses from essential matrix
                poses = ExtractCameraPose(E)
                
                # First camera is at the origin with identity rotation
                C1 = np.zeros(3)
                R1 = np.eye(3)
                
                # Triangulate points for all four poses
                X_set = []
                for pose_idx, (C2, R2) in enumerate(poses):
                    X = LinearTriangulation(K, C1, R1, C2, R2, inlier_pts1, inlier_pts2)
                    X_set.append(X)
                
                # Disambiguate camera pose using cheirality condition
                C_best, R_best, X_best, best_pose_idx = DisambiguateCameraPose(poses, K, inlier_pts1, inlier_pts2, X_set)
                visualize_disambiguated_pose(poses, X_set, best_pose_idx, Output)
                
                print(f"Best camera pose between images {i+1} and {j+1}:")
                print(f"Camera center: {C_best}")
                print(f"Rotation matrix: {R_best}")
                print(f"Number of triangulated points: {X_best.shape[0]}")
                
                # Refine 3D points using non-linear triangulation
                X_refined = NonlinearTriangulation(K, C1, R1, C_best, R_best, inlier_pts1, inlier_pts2, X_best)
                visualize_linear_nonlinear_comparison(X_best, X_refined, C1, R1, C_best, R_best, Output)
                
                # Store the best camera pose and 3D points
                camera_poses[(i+1, j+1)] = (C_best, R_best)
                points_3d[(i+1, j+1)] = X_refined
                
                # Visualize triangulation comparison
                visualize_triangulation_comparison(
                    [images[i], images[j]], 
                    inlier_pts1, inlier_pts2, 
                    K, C1, R1, C_best, R_best, 
                    X_best, X_refined, 
                    Output, f"{i+1}_{j+1}"
                )
                visualize_initial_triangulation(poses, X_set, Output)
    
    # Initialize camera poses for the full reconstruction
    all_camera_poses = {1: (np.zeros(3), np.eye(3))}  # First camera at origin
    all_points_3d = np.zeros((0, 3))  # Empty array to store all 3D points
    
    # Add second camera from the first pair (1,2)
    all_camera_poses[2] = camera_poses[(1, 2)]
    
    # Add initial 3D points
    all_points_3d = np.vstack((all_points_3d, points_3d[(1, 2)]))
    
    # Initialize point tracking arrays
    X_all = np.zeros((feature_x.shape[0], 3))  # Store all 3D points
    X_found = np.zeros((feature_x.shape[0], 1), dtype=int)  # Track which points are triangulated
    
    # Mark points from initial triangulation as found
    idx = np.where(filtered_feature_flag[:, 0] & filtered_feature_flag[:, 1])[0]
    if len(idx) == points_3d[(1, 2)].shape[0]:
        X_all[idx] = points_3d[(1, 2)]
        X_found[idx] = 1
    else:
        print("Warning: Shape mismatch in initial triangulation")
        min_len = min(len(idx), points_3d[(1, 2)].shape[0])
        X_all[idx[:min_len]] = points_3d[(1, 2)][:min_len]
        X_found[idx[:min_len]] = 1
    
    # Register remaining cameras using PnP
    for k in range(3, 6):  # Images 3, 4, 5
        print(f"\nRegistering camera {k}...")
        
        # Find 2D-3D correspondences
        visible_points = np.where(filtered_feature_flag[:, k-1] & X_found.flatten())[0]
        
        if len(visible_points) < 6:
            print(f"Not enough correspondences for image {k}")
            continue
        
        # Get 3D points and their 2D projections in the new image
        X_3d = X_all[visible_points]
        x_2d = np.hstack((feature_x[visible_points, k-1].reshape(-1, 1), 
                          feature_y[visible_points, k-1].reshape(-1, 1)))
        
        # Estimate camera pose using PnP RANSAC
        C_new, R_new, inliers = PnPRANSAC(X_3d, x_2d, K)

        linear_error = calculate_reprojection_error(X_3d[inliers], x_2d[inliers], K, C_new, R_new)
        print(f"Linear PnP - Mean reprojection error: {linear_error:.6f}px")
        
        if C_new is None or R_new is None or len(inliers) < 6:
            print(f"PnP RANSAC failed for camera {k}")
            continue
            
        print(f"PnP RANSAC found {len(inliers)} inliers out of {len(X_3d)} points")

        
        # Refine camera pose using non-linear PnP
        C_refined, R_refined = NonlinearPnP(X_3d[inliers], x_2d[inliers], K, C_new, R_new)

        nonlinear_error = calculate_reprojection_error(X_3d[inliers], x_2d[inliers], K, C_refined, R_refined)
        print(f"Non-linear PnP - Mean reprojection error: {nonlinear_error:.6f}px")
        
        # Store the new camera pose
        all_camera_poses[k] = (C_refined, R_refined)

        all_points = np.vstack([points for points in points_3d.values()])

        # Visualize camera poses after adding each new camera
        visualize_camera_poses(all_camera_poses, all_points, Output, f"sfm_views_1_to_{k}.png")
        
        # Triangulate new points with previous cameras
        for prev_k in range(1, k):
            # Find correspondences between current and previous images
            idx = np.where(filtered_feature_flag[:, prev_k-1] & filtered_feature_flag[:, k-1])[0]
            
            if len(idx) < 8:
                continue
            
            # Get 2D points
            pts_prev = np.hstack((feature_x[idx, prev_k-1].reshape(-1, 1), 
                                 feature_y[idx, prev_k-1].reshape(-1, 1)))
            pts_curr = np.hstack((feature_x[idx, k-1].reshape(-1, 1), 
                                 feature_y[idx, k-1].reshape(-1, 1)))
            
            # Get camera poses
            C_prev, R_prev = all_camera_poses[prev_k]
            
            # Triangulate points
            X_new = LinearTriangulation(K, C_prev, R_prev, C_refined, R_refined, pts_prev, pts_curr)
            
            # Refine points using non-linear triangulation
            X_new_refined = NonlinearTriangulation(K, C_prev, R_prev, C_refined, R_refined, 
                                                  pts_prev, pts_curr, X_new)
            
            # Update point tracking arrays
            if len(idx) == X_new_refined.shape[0]:
                X_all[idx] = X_new_refined
                X_found[idx] = 1
            else:
                print(f"Warning: Shape mismatch in triangulation. idx: {len(idx)}, points: {X_new_refined.shape[0]}")
                min_len = min(len(idx), X_new_refined.shape[0])
                X_all[idx[:min_len]] = X_new_refined[:min_len]
                X_found[idx[:min_len]] = 1
            
            # Add new points to the reconstruction
            all_points_3d = np.vstack((all_points_3d, X_new_refined))
        
        # Build visibility matrix for current set of cameras
        V, _ = BuildVisibilityMatrix(filtered_feature_flag[:, :k], X_found)
        
        # Perform bundle adjustment after each new camera
        print(f"Performing bundle adjustment after registering camera {k}...")
        points_3d_before_BA = X_all[X_found.flatten() > 0].copy()

        visibility_matrix = validate_bundle_adjustment_inputs(all_camera_poses, points_3d_before_BA, V)
        refined_camera_poses, refined_points_3d = BundleAdjustment(all_camera_poses, points_3d_before_BA, K, visibility_matrix)
        
        # refined_camera_poses, refined_points_3d = BundleAdjustment(all_camera_poses, points_3d_before_BA, K, V)
        
        # Update camera poses and 3D points with refined values
        all_camera_poses = refined_camera_poses
        
        # Update X_all with refined points
        X_all[X_found.flatten() > 0] = refined_points_3d
    
    # Final bundle adjustment on all cameras and points
    print("\nPerforming final bundle adjustment...")
    V, _ = BuildVisibilityMatrix(filtered_feature_flag, X_found)
    
    points_3d_before_BA = X_all[X_found.flatten() > 0].copy()

    visibility_matrix = validate_bundle_adjustment_inputs(all_camera_poses, points_3d_before_BA, V)
    refined_camera_poses, refined_points_3d = BundleAdjustment(all_camera_poses, points_3d_before_BA, K, visibility_matrix)
    # refined_camera_poses, refined_points_3d = BundleAdjustment(all_camera_poses, points_3d_before_BA, K, V)
    
        # Visualize final reconstruction
    visualize_final_reconstruction(refined_camera_poses, refined_points_3d, points_3d_before_BA, Output)
    
    print("Processing complete. Results saved to", Output)

if __name__ == "__main__":
    main()

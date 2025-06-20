# Structure from Motion (SfM) Pipeline

This repository implements a **Structure from Motion (SfM)** pipeline for 3D reconstruction from multiple images. The pipeline includes feature extraction, camera pose estimation, triangulation, bundle adjustment, and visualization of the reconstructed 3D scene.


## Directory Structure


YourProject/
├── README.md                              # This file
├── Phase1/                                # Sparse reconstruction code
│   ├── GetInliersRANSAC.py                # RANSAC-based inlier detection
│   ├── EstimateFundamentalMatrix.py       # Fundamental matrix estimation
│   ├── EssentialMatrixFromFundamentalMatrix.py # Essential matrix computation
│   ├── ExtractCameraPose.py               # Extracts camera poses from essential matrix
│   ├── LinearTriangulation.py             # Linear triangulation of 3D points
│   ├── DisambiguateCameraPose.py          # Disambiguates correct camera pose
│   ├── NonlinearTriangulation.py          # Refines 3D points using non-linear optimization
│   ├── PnPRANSAC.py                       # PnP with RANSAC for camera registration
│   ├── NonlinearPnP.py                    # Refines camera poses using non-linear optimization
│   ├── BuildVisibilityMatrix.py           # Builds visibility matrix for bundle adjustment
│   ├── BundleAdjustment.py                # Optimizes camera poses and/or 3D points
│   └── Wrapper.py                         # Main pipeline script for Phase 1
├── Data/                                  # Input data directory (images, calibration, etc.)
│   ├── IntermediateOutputImages/         # Intermediate outputs (optional)
├── Outputs/                               # Output directory for results
└── Report.pdf                             # Project report describing implementation details



## Features

### Phase 1: Sparse Reconstruction Pipeline

1. **Feature Extraction**:
   - Extracts features from matching files (`matching*.txt`) and computes RGB values.
2. **Fundamental Matrix Estimation**:
   - Computes the fundamental matrix using RANSAC to reject outliers.
3. **Essential Matrix Computation**:
   - Converts the fundamental matrix into an essential matrix using the calibration matrix.
4. **Camera Pose Estimation**:
   - Extracts four possible camera poses from the essential matrix.
5. **Triangulation**:
   - Performs linear triangulation to compute 3D points.
6. **Pose Disambiguation**:
   - Chooses the correct camera pose using the cheirality condition.
7. **Non-linear Triangulation**:
   - Refines the computed 3D points using non-linear optimization.
8. **Camera Registration**:
   - Registers additional cameras using PnP with RANSAC and refines their poses.
9. **Bundle Adjustment**:
   - Optimizes camera poses and/or 3D points jointly to minimize reprojection error.
10. **Visualization**:
    - Generates visualizations of matches, epipolar lines, triangulated points, and final reconstruction.


### Prerequisites

1. Python >= 3.8
2. Required Python libraries:
```bash
pip install numpy matplotlib opencv-python scipy argparse
```

## Usage

### Input Data Requirements

1. Place input images (`1.png`, `2.png`, ..., `5.png`) in the `Data/` folder.
2. Include a `calibration.txt` file in the `Data/` folder containing the intrinsic calibration matrix.

### Run Sparse Reconstruction (Phase 1)

1. Execute the main pipeline script:

```bash
python Wrapper.py --Data ./Data --Outputs ./Outputs/
```
or 

```bash
python Wrapper.py
```

2. Outputs will be saved in the `Outputs/` directory, including:
    - Matches before/after RANSAC (`matches_before_ransac_*.png`, `matches_after_ransac_*.png`)
    - Epipolar lines (`epipolar_lines_*.png`)
    - Triangulated points (`initial_triangulation.png`, `disambiguated_pose.png`)
    - Final reconstruction (`reconstruction_3d.png`, `reconstruction_top_view.png`, etc.)



## Code Workflow

### Feature Extraction (`features_extraction`)
- Reads feature matches from `matching*.txt` files and stores feature coordinates, visibility flags, and RGB values.

### Fundamental Matrix Estimation (`GetInliersRANSAC`)
- Estimates the fundamental matrix between image pairs using RANSAC to reject outliers.

### Essential Matrix Computation (`EssentialMatrixFromFundamentalMatrix`)
- Converts the estimated fundamental matrix into an essential matrix using intrinsic calibration.

### Camera Pose Estimation (`ExtractCameraPose`)
- Extracts four possible camera poses (rotation and translation) from the essential matrix.

### Triangulation (`LinearTriangulation` and `NonlinearTriangulation`)
- Performs linear triangulation to compute initial 3D point estimates.
- Refines these points using non-linear optimization to minimize reprojection error.

### Pose Disambiguation (`DisambiguateCameraPose`)
- Chooses the correct camera pose based on cheirality constraints (ensuring all points are in front of both cameras).

### Camera Registration (`PnPRANSAC` and `NonlinearPnP`)
- Registers additional cameras by solving Perspective-n-Point (PnP) problems with RANSAC to handle outliers.
- Refines camera poses using non-linear optimization.

### Bundle Adjustment (`BundleAdjustment`)
- Optimizes all camera poses and/or 3D points jointly to minimize reprojection error.

### Visualization Functions
- Visualizes matches before/after RANSAC filtering.
- Draws epipolar lines on image pairs.
- Compares linear vs non-linear triangulation results.
- Visualizes final reconstruction in both 3D space and top-down views.


## Outputs

The pipeline generates several outputs during execution:

1. **Matches Before/After RANSAC**: 
    - Visualizes feature matches before and after RANSAC filtering.
2. **Epipolar Lines**: 
    - Visualizes epipolar constraints between image pairs.
3. **Triangulated Points**: 
    - Displays initial and refined triangulated points in both linear and non-linear triangulation.
4. **Final Reconstruction**: 
    - Shows all registered cameras and reconstructed points in a top-down view (`reconstruction_top_view.png`) or a full 3D view (`reconstruction_3d.png`).



#### Matches Before/After RANSAC:
Matches Before
Matches After

#### Epipolar Lines:
Epipolar Lines

#### Final Reconstruction (Top View):
Top View

#### Final Reconstruction (3D View):
3D View

# 📸 Structure from Motion (SfM) Sparse 3D Reconstruction

This project implements a **Structure from Motion (SfM)** pipeline to estimate camera poses and reconstruct a 3D scene from a set of 2D images. It covers key components of multiview geometry including fundamental matrix estimation, essential matrix computation, triangulation, PnP, and bundle adjustment.

---

## 🧩 Pipeline Overview

The pipeline is implemented in `Code/Wrapper.py` and consists of the following major steps:

1. **Feature Extraction**  
   Extracts precomputed 2D feature correspondences from `matching*.txt` files.

2. **Fundamental Matrix Estimation + RANSAC**  
   Estimates the epipolar geometry between two views with outlier rejection.
   - `EstimateFundamentalMatrix.py`
   - `GetInliersRANSAC.py`

3. **Essential Matrix Computation**  
   Computes the essential matrix from the fundamental matrix and intrinsic matrix.
   - `EssentialMatrixFromFundamentalMatrix.py`

4. **Camera Pose Extraction**  
   Extracts the four possible relative camera poses from the essential matrix.
   - `ExtractCameraPose.py`

5. **Triangulation**
   - **Linear Triangulation:** Estimates 3D points from image correspondences.
     - `LinearTriangulation.py`
   - **Nonlinear Triangulation:** Refines 3D points by minimizing reprojection error.
     - `NonlinearTriangulation.py`

6. **Camera Pose Estimation (PnP)**
   - **PnP with RANSAC:** Estimates camera poses using 2D-3D correspondences.
     - `PnPRANSAC.py`
   - **Nonlinear PnP:** Refines camera poses.
     - `NonlinearPnP.py`

7. **Bundle Adjustment**  
   Optimizes all camera poses and 3D points jointly for global consistency.
   - `BuildVisibilityMatrix.py`
   - `BundleAdjustment.py`

8. **Visualization & Output**  
   Generates 3D visualizations of the reconstructed scene and camera poses.

---

## 🗂 Project Structure

```

YourProject/
├── README.md
├── Code/
│   ├── Wrapper.py                        # Main pipeline
│   ├── EstimateFundamentalMatrix.py     # 8-point algorithm
│   ├── GetInliersRANSAC.py              # RANSAC for robust F estimation
│   ├── EssentialMatrixFromFundamentalMatrix.py
│   ├── ExtractCameraPose.py
│   ├── LinearTriangulation.py
│   ├── NonlinearTriangulation.py
│   ├── PnPRANSAC.py
│   ├── NonlinearPnP.py
│   ├── BuildVisibilityMatrix.py
│   ├── BundleAdjustment.py
├── Data/                                 # Input images, calibration, matching
│   └── IntermediateOutputImages/        # Optional debug outputs
├── Outputs/                              # Visualizations and 3D results
└── Report.pdf                            # Full methodology and results

````

---

## 🧪 Sample Results

- **Epipolar Lines Visualization**  
  Epipolar geometry validated using the fundamental matrix.

- **Initial and Refined 3D Points**  
  Shows improvement using nonlinear triangulation.

- **Top-Down and Side Views of Camera Poses**  
  Visualizes the full reconstruction using all camera views.

---

## 🛠 Requirements

- Python 3.7+
- NumPy
- OpenCV
- Matplotlib
- SciPy

```bash
pip install numpy opencv-python matplotlib scipy
````

---

## 🚀 Running the Code

1. Place your input data (e.g., `matching*.txt`, calibration file) in the `Data/` directory.
2. Make sure all `.py` files are in the `Code/` directory.
3. Run the main script:

```bash
cd Code
python Wrapper.py
```

This will generate visual outputs in the `Outputs/` folder.

---

## 🧑‍💻 Contributors

* **Pavan Ganesh Pabbineedi** - [ppabbineedi@wpi.edu](mailto:ppabbineedi@wpi.edu)
* **Manideep Duggi** - [mduggi@wpi.edu](mailto:mduggi@wpi.edu)

---

## 📄 Reference

For a detailed explanation of the theory, implementation, and results, see the [Report.pdf](./Report.pdf) included in the repository.

---

## 🏛 Acknowledgment

This project was conducted as part of the **Robotics Engineering Program at Worcester Polytechnic Institute (WPI)**.



# Harris Corner Eigen Analysis 📐
This repository contains a classical computer vision implementation of the Harris Corner Detection algorithm, extended with structure tensor eigenvalue and eigenvector visualization.

## Overview
The project computes image gradients using Sobel operators, constructs the Harris structure tensor, and detects corner points based on the Harris response function.  
For each detected corner, the eigenvalues and eigenvectors of the structure tensor are calculated and visualized to illustrate local intensity variation directions.

## Methodology
1. Convert input image to grayscale  
2. Compute image gradients (Sobel X and Y)  
3. Construct structure tensor components  
4. Apply Gaussian smoothing  
5. Compute Harris response function  
6. Threshold and detect corner points  
7. Compute eigenvalues and eigenvectors at each corner  
8. Visualize eigenvectors as directional arrows  

## Outputs
- **corners.jpg** – detected Harris corner points  
- **eigenvectors.jpg** – eigenvector visualization at each corner  

## Requirements
- Python 3.x  
- OpenCV  
- NumPy  

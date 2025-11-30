# Sudoku Solver with Computer Vision

This Project was made in July 2023, this is an intelligent Sudoku solver that uses computer vision and deep learning to extract puzzles from images and solve them automatically. The project combines OpenCV for image processing, TensorFlow/Keras for digit recognition, and a backtracking algorithm for solving.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Webcam Capture**: Real-time capture of Sudoku puzzles using your webcam
- **Automatic Grid Detection**: Advanced computer vision to detect and extract the Sudoku grid from images
- **Deep Learning OCR**: CNN-based digit recognition trained on MNIST dataset (more accurate than traditional OCR)
- **Intelligent Solver**: Backtracking algorithm that solves any valid Sudoku puzzle
- **Visual Overlay**: Displays the solution overlaid on the original image
- **Smart Cleanup**: Automatically removes OCR errors that violate Sudoku rules

## How It Works

### 1. Image Processing Pipeline
- **Preprocessing**: Converts image to grayscale and applies adaptive thresholding
- **Grid Detection**: Finds the largest quadrilateral contour (Sudoku grid)
- **Perspective Transform**: Warps the image to a top-down view
- **Cell Extraction**: Divides the grid into 81 individual cells

### 2. Digit Recognition
- **Preprocessing**: Each cell is processed (blur, threshold, contour extraction)
- **CNN Classification**: Uses a trained convolutional neural network to recognize digits
- **Confidence Filtering**: Only accepts predictions with >80% confidence
- **Error Correction**: Removes digits that violate Sudoku rules

### 3. Solving Algorithm
- **Backtracking**: Recursive depth-first search algorithm
- **Constraint Checking**: Validates row, column, and 3x3 box constraints
- **Solution Overlay**: Projects the solution back onto the original image

## Model Architecture

The digit recognition model is a simple CNN trained on the MNIST dataset:

```
Input: 28x28 grayscale image
├── Conv2D(32 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(64 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Flatten
├── Dense(64) + ReLU
└── Dense(10) + Softmax (output: digit 0-9)
```

**Training**: 3 epochs on MNIST with 90% training / 10% validation split

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

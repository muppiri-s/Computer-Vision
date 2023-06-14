# Computer-Vision

***Image Processing Assignments***
This repository contains three assignments for the Image Processing course.

# Assignment 1: Laplacian Blending
In this assignment, Laplacian blending techniques are used to blend two images together.

Input
The input to this assignment is a set of three images: an apple image, a mask image, and an orange image.

Procedure
1. Crop and resize the three input images to the same size.
2. Construct a Gaussian pyramid for each of the three images.
3. Create a Laplacian image for each of the apple and orange Gaussian pyramids.
4. Blend the two Laplacian images using the mask image.
5. Reconstruct the blended image.

# Assignment 2: Canny Edge Detection
In this assignment, implemented the Canny edge detection algorithm.

Input
The input to this assignment is an image.

Procedure
1. Apply a Gaussian blur to the image.
2. Calculate the gradient of the image.
3. Apply non-maximum suppression to the gradient magnitude.
4. Apply hysteresis thresholding to the gradient magnitude.
5. Identify edges in the image.

# Assignment 3: Hough Transform
In this assignment, Hough transform to detect lines in an image.

Input
The input to this assignment is an image.

Procedure
1. Convert the image to grayscale.
2. Apply a threshold to the grayscale image.
3. Find the edge points in the image.
4. For each edge point, create a line segment.
5. Use the Hough transform to vote for each line segment.
6. Identify the lines with the most votes.

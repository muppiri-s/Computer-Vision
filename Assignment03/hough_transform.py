from matplotlib import pyplot as plt
import numpy as np
import math
import cv2

def hough_line(edge):
    # Theta ranging from 0 - 180 degree
    theta = np.arange(0, 180, 1)
    cos = np.cos(np.deg2rad(theta))
    sin = np.sin(np.deg2rad(theta))

    # Generate a accumulator matrix to store the values
    rho_range = round(math.sqrt(edge.shape[0]**2 + edge.shape[1]**2))
    accumulator = np.zeros((2 * rho_range, len(theta)), dtype=np.uint8)

    # Get threshold to get edges pixel location along x, y
    edge_pixels = np.where(edge == 255)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

    # Determine the rho value for the edge location along x, y with all the theta range
    for p in range(len(coordinates)):
        for t in range(len(theta)):
            rho = int(round(coordinates[p][1] * cos[t] + coordinates[p][0] * sin[t]))
            accumulator[rho, t] += 2 # Suppose add 1 only, Just want to get clear result

    return accumulator

# Read image, Conver to grey scale, Find the edges using canny edge detection
image = cv2.imread('image4.png')
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
edges = cv2.Canny(grayscale,50, 200, None, 3)

# Calculate and perform the hough line detection 
accumulator = hough_line(edges)

# Set the threshold and draw the lines for the same image
edge_pixels = np.where(accumulator > 110)
coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

# Line equation is used to draw the lines which are detected on the original image
for i in range(0, len(coordinates)):
    a = np.cos(np.deg2rad(coordinates[i][1]))
    b = np.sin(np.deg2rad(coordinates[i][1]))
    x0 = a*coordinates[i][0]
    y0 = b*coordinates[i][0]
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),1)
# display the result

plt.title('Hough Line'), plt.imshow(image)
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit

def compute_values(threshold, sigma):
    temp = -np.log(threshold) * 2 * (sigma ** 2)
    return np.round(np.sqrt(temp))

# Generate the mask for the image. 
def generate_mask(threshold, sigma):
    values = compute_values(threshold, sigma)
    y, x = np.meshgrid(range(-int(values), int(values) + 1), range(-int(values), int(values) + 1))
    return x, y

def gaussian_filter(x,y, sigma):
    temp = ((x ** 2) + (y ** 2)) / (2 * (sigma ** 2))
    return (np.exp(-temp))

# Calculate gradient mask in x-direction
def calculate_gradient_X(x,y, sigma):
    temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    return -((x * np.exp(-temp)) / sigma ** 2)

# Calculate gradient mask in y-direction 
def calculate_gradient_Y(x,y, sigma):
    temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    return -((y * np.exp(-temp)) / sigma ** 2)

def padding(img, kernel):
    rows, cols = img.shape
    krows, kcols = kernel.shape
    padded = np.zeros((rows + krows,cols + kcols), dtype=img.dtype)
    insert = np.uint((krows)/2)
    padded[insert: insert + rows, insert: insert + cols] = img
    return padded
            
# Smooth the image to reduce the intensity in pixels
def noise_reduction(img, kernel):
    mask = kernel
    i, j = mask.shape
    output = np.zeros((img.shape[0], img.shape[1]))           
    image_padded = padding(img, mask)
    for x in range(img.shape[0]):    
        for y in range(img.shape[1]):
            output[x, y] = (mask * image_padded[x:x+i, y:y+j]).sum() / mask.sum()  
    return output

# Create gradient mask in x and y direction
def generate_gradient_x(fx, fy):
    gx = calculate_gradient_X(fx, fy, sigma)
    gx = (gx * 255)
    return np.around(gx)

def generate_gradient_y(fx, fy):    
    gy = calculate_gradient_Y(fx, fy, sigma)
    gy = (gy * 255)
    return np.around(gy)

def apply_mask(image, kernel):
    i, j = kernel.shape
    kernel = np.flipud(np.fliplr(kernel))    
    output = np.zeros_like(image)           
    image_padded = padding(image, kernel)
    for x in range(image.shape[0]):    
        for y in range(image.shape[1]):
            output[x, y] = (kernel * image_padded[x:x+i, y:y+j]).sum()        
    return output

def grad_magnitude(fx, fy):
    magnitude = np.zeros((fx.shape[0], fx.shape[1]))
    magnitude = np.sqrt((fx ** 2) + (fy ** 2))
    magnitude = magnitude * 100 / magnitude.max()
    return np.around(magnitude)

def grad_orientation(fx, fy):
    grad_or = np.zeros((fx.shape[0], fx.shape[1]))
    grad_or = np.rad2deg(np.arctan2(fy, fx)) + 180
    return grad_or

def grad_quantization(angle):
    quantized = np.zeros((angle.shape[0], angle.shape[1]))
    for i in range(angle.shape[0]):
        for j in range(angle.shape[1]):
            if 0 <= angle[i, j] <= 22.5 or 157.5 <= angle[i, j] <= 202.5 or 337.5 < angle[i, j] < 360:
                quantized[i, j] = 0
            elif 22.5 <= angle[i, j] <= 67.5 or 202.5 <= angle[i, j] <= 247.5:
                quantized[i, j] = 1
            elif 67.5 <= angle[i, j] <= 122.5 or 247.5 <= angle[i, j] <= 292.5:
                quantized[i, j] = 2
            elif 112.5 <= angle[i, j] <= 157.5 or 292.5 <= angle[i, j] <= 337.5:
                quantized[i, j] = 3
    return quantized
    
def non_max_sup(quant, magnitude, D):
    nms = np.zeros(quant.shape)
    a, b = np.shape(quant)
    for i in range(a-1):
        for j in range(b-1):
            if quant[i,j] == 0:
                if  magnitude[i,j-1]< magnitude[i,j] or magnitude[i,j] > magnitude[i,j+1]:
                    nms[i,j] = D[i,j]
                else:
                    nms[i,j] = 0
            if quant[i,j]==1:
                if  magnitude[i-1,j+1]<= magnitude[i,j] or magnitude[i,j] >= magnitude[i+1,j-1]:
                    nms[i,j] = D[i,j]
                else:
                    nms[i,j] = 0       
            if quant[i,j] == 2:
                if  magnitude[i-1,j]<= magnitude[i,j] or magnitude[i,j] >= magnitude[i+1,j]:
                    nms[i,j] = D[i,j]
                else:
                    nms[i,j] = 0
            if quant[i,j] == 3:
                if  magnitude[i-1,j-1]<= magnitude[i,j] or magnitude[i,j] >= magnitude[i+1,j+1]:
                    nms[i,j] = D[i,j]
                else:
                    nms[i,j] = 0
    return nms

def double_thresholding(g_suppressed, low_threshold, high_threshold):
    g_thresholded = np.zeros(g_suppressed.shape)
    # loop over pixels
    for i in range(0, g_suppressed.shape[0]):		
        for j in range(0, g_suppressed.shape[1]):
            if g_suppressed[i,j] < low_threshold:
                g_thresholded[i,j] = 0
            elif g_suppressed[i,j] >= low_threshold and g_suppressed[i,j] < high_threshold: 	
                g_thresholded[i,j] = 128
            else:					        
                g_thresholded[i,j] = 255
    return g_thresholded

def hysteresis(g_thresholded):
    g_strong = np.zeros(g_thresholded.shape)
    # loop over pixels
    for i in range(0, g_thresholded.shape[0]):		
        for j in range(0, g_thresholded.shape[1]):
            val = g_thresholded[i,j]
            if val == 128:			
                if g_thresholded[i-1,j] == 255 or g_thresholded[i+1,j] == 255 or g_thresholded[i-1,j-1] == 255 or g_thresholded[i+1,j-1] == 255 or g_thresholded[i-1,j+1] == 255 or g_thresholded[i+1,j+1] == 255 or g_thresholded[i,j-1] == 255 or g_thresholded[i,j+1] == 255:
                    g_strong[i,j] = 255		
            elif val == 255:
                g_strong[i,j] = 255		
    return g_strong

start = timeit.default_timer()

sigma = 5
threshold = 0.1

x, y = generate_mask(threshold, sigma)
gauss = gaussian_filter(x, y, sigma)

gradient_x = -generate_gradient_x(x, y)
gradient_y = -generate_gradient_y(x, y)

# Read the image and convert to gray scale
image = cv2.imread('img2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Reduce the intensity in the pixels
smooth_image = noise_reduction(gray, gauss)

# Apple the x and y gradient masks to the image
fx = apply_mask(smooth_image, gradient_x)
fy = apply_mask(smooth_image, gradient_y)

# Calculate the gradient magnitude at every pixel location 
gradient_magnitude = grad_magnitude(fx, fy)
gradient_magnitude = gradient_magnitude.astype(int)

# Calculate orientation of gradient
gradient_orientation = grad_orientation(fx, fy)

# Quantaization of gradients to 4 directions
gradient_quantization = grad_quantization(gradient_orientation)
non_maximum_supresion = non_max_sup(gradient_quantization, gradient_orientation, gradient_magnitude)

# Double thresholding to follow edges
threshold = double_thresholding(non_maximum_supresion, 10, 10)
hysteresis_thresholding = hysteresis(threshold)

stop = timeit.default_timer()

# Display the output
plt.figure(figsize = (10,10))
plt.imshow(hysteresis_thresholding, cmap='gray')
plt.show()
print('Time Elapsed: ', stop - start)  

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# import timeit

# start = timeit.default_timer()

# img = cv.imread('image1.jpg', cv.IMREAD_GRAYSCALE)
# edges = cv.Canny(img, 100, 200)

# stop = timeit.default_timer()

# plt.subplot(121), plt.imshow(img, cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])

# plt.subplot(121), plt.imshow(edges, cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# plt.show()
# print('Time Elapsed: ', stop - start)  
# Image Blending
import cv2 as cv
import numpy as np,sys

apple = cv.imread('C:/Users/sahit/OneDrive/Desktop/CS_grad/CV/apple1.png')
orange = cv.imread('C:/Users/sahit/OneDrive/Desktop/CS_grad/CV/orange1.png')
mask = cv.imread('C:/Users/sahit/OneDrive/Desktop/ComputerVision/mask.png')

print("APPLE: [rows, columns, channel]", apple.shape)
print("ORANGE: [rows, columns, channel]",orange.shape)
print("Mask: [rows, columns, channel]", mask.shape)

# #merge half of apple & orange images
# apple_orange = np.hstack((apple[:, :206], orange[:, 206:]))

# #Generate gaussian pyramid for apple
# apple_copy = apple.copy()
# gp_apple = [apple_copy]
# for i in range(5):
#     apple_copy = cv.pyrDown(apple_copy)
#     gp_apple.append(apple_copy)

# #Generate gaussian pyramid for orange
# orange_copy = orange.copy()
# gp_orange = [orange_copy]
# for i in range(5):
#     orange_copy = cv.pyrDown(orange_copy)
#     gp_orange.append(orange_copy)

# #Generate gaussian pyramid for mask
# mask_copy = mask.copy()
# gp_mask = [mask_copy]
# for i in range(5):
#     mask_copy = cv.pyrDown(mask_copy)
#     gp_mask.append(np.float32(mask_copy))

# #Calculate Laplacian pyramid for apple
# apple_copy = gp_apple[5]
# lp_apple = [apple_copy]
# for i in range(5, 0, -1):
#     size = (gp_apple[i-1].shape[1], gp_apple[i-1].shape[0])
#     gaussian_expanded = cv.pyrUp(gp_apple[i], dstsize=size)
#     laplacian = cv.subtract(gp_apple[i-1], gaussian_expanded)
#     lp_apple.append(laplacian)

# #Calculate Laplacian pyramid for orange
# orange_copy = gp_orange[5]
# lp_orange = [orange_copy]
# for i in range(5, 0, -1):
#     size = (gp_apple[i-1].shape[1], gp_apple[i-1].shape[0])
#     gaussian_expanded = cv.pyrUp(gp_orange[i], dstsize=size)
#     laplacian = cv.subtract(gp_orange[i-1], gaussian_expanded)
#     lp_orange.append(laplacian)

# #Blend two images according to the mask
# apple_orange_pyramid = []
# gp_mask.reverse()
# for apple_lap, orange_lap, mask in zip(lp_apple, lp_orange, gp_mask):
#     ap_or = orange_lap * mask + apple_lap * (1.0 - mask)
#     apple_orange_pyramid.append(ap_or)
#     cv.imshow(str(i), ap_or)
#     cv.waitKey(1000)

# # #reconstruct image
# laplacian_top = apple_orange_pyramid[0]
# for i in range(0, 5):
#     #size = (apple_orange_pyramid[i].shape[1], apple_orange_pyramid[i].shape[0])
#     laplacian_expanded = cv.pyrUp(laplacian_top)
#     laplacian_expanded = cv.add(laplacian_top, apple_orange_pyramid[i])
#     cv.imshow(str(i), laplacian_expanded)
#     cv.waitKey(1000)

# # final = reconstruct()
# # cv.imwrite('C:/Users/sahit/OneDrive/Desktop/CS_grad/CV/final.png', final[5])
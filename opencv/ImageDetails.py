import cv2
# reading the image
image = cv2.imread('cv.png')
# printing the details of the image
print(image.dtype)
print(image.shape)
print(image.size)
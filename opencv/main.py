# importing cv2
import cv2
# reading the image
image = cv2.imread('cv.png')
# showing the image
cv2.imshow('OpenCv',image)
# rewriting the image
cv2.imwrite('abcd.png',image)
# wait is till until we close the window 
cv2.waitKey(0)
# close all the windows
cv2.destroyAllWindows()
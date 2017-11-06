import cv2
import  numpy as np 

img = cv2.imread('/Users/Shared/OpenCV_homework/Test_images/baboon.jpg')
cv2.namedWindow("Image") 
cv2.imshow("Image",img)

print("RGB value: ",img[20,25])

#img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
b,g,r = cv2.split(img)

cv2.namedWindow("Blue")
cv2.imshow("Blue",b)
cv2.namedWindow("Green")
cv2.imshow("Green",g)
cv2.namedWindow("Red")
cv2.imshow("Red",r)


img2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

print("YCrCb value: ",img2[20,25])

Y,Cb,Cr = cv2.split(img2)
cv2.namedWindow("Y")
cv2.imshow("Y",Y)
cv2.namedWindow("Cb")
cv2.imshow("Cb",Cb)
cv2.namedWindow("Cr")
cv2.imshow("Cr",Cr)

img3 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print("HSV value: ",img3[20,25])

H,S,V = cv2.split(img3)
cv2.namedWindow("H")
cv2.imshow("H",H)
cv2.namedWindow("S")
cv2.imshow("S",S)
cv2.namedWindow("V")
cv2.imshow("V",V)

cv2.waitKey(0)
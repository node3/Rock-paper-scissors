# USAGE
# python ImagePreprocessing.py --image IMG.jpg

# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(h, w, d) = image.shape
cv2.imshow("Image", image)
cv2.waitKey(0)

r = 300.0 / w
dim = (300, int(h * r))
image = cv2.resize(image, dim)
cv2.imshow("Aspect Ratio Resize", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAY", gray)
cv2.waitKey(0)

blur = cv2.GaussianBlur(gray,(5,5),2)
cv2.imshow("GaussianBlur", blur)
cv2.waitKey(0)

th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
cv2.imshow("adaptiveThreshold", th3)
cv2.waitKey(0)

ret, res = cv2.threshold(th3, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("OTSU", res)
cv2.waitKey(0)
# USAGE
# python ImagePreprosessing.py --image img_src/rock/

def image_resize(image, width = None, height = None):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # return the resized image
    return resized

def transformImage(image, dim_h):

    dim = (dim_h, dim_h)
    image = cv2.resize(image, dim)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    return res


def rotate_bound(image, angle):

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255))

def resizeROI(image, dim):
    # returns dim*dim image from center of 'image'
    (h, w) = image.shape
    cx, cy = h/2, w/2
    x1, y1 = int(cx - dim[0]/2), int(cy - dim[1]/2)
    x2, y2 = int(cx + dim[0]/2), int(cy + dim[1]/2)
    roi = image[y1:y2, x1:x2]
    return roi

def augmentHelper(image):
    images = []

    #Flip
    vertical_img = cv2.flip( image, 1)

    for angle in np.arange(0, 360, 30):
        rotatedImg = rotate_bound(image, angle)
        resizedImg = resizeROI(rotatedImg, image.shape)
        images.append(resizedImg.copy())

    for angle in np.arange(0, 360, 30):
        rotatedImg = rotate_bound(vertical_img, angle)
        resizedImg = resizeROI(rotatedImg, image.shape)
        images.append(resizedImg.copy())

    return images

def augmentTranslate(image, x, y):
    r,c = image.shape
    M = np.float32([[1,0,x],[0,1,y]])
    return cv2.warpAffine(image, M, (c,r), borderValue=(255,255,255))


def augmentImage(image):
    images = []

    (h, w) = image.shape
    cx, cy = h/2, w/2
    translatedImage = []

    for shift_x in np.arange(0, cx - 40, 20):
        translatedImage.append(augmentTranslate(image, shift_x, 0))

    for shift_y in np.arange(10, cy - 40, 20):
        translatedImage.append(augmentTranslate(image, 0, shift_y))

    for shift_x, shift_y in zip(np.arange(10, cx - 40, 20), np.arange(0, cy - 40, 20)):
        translatedImage.append(augmentTranslate(image, shift_x, shift_y))


    for img in translatedImage:
        images.extend(augmentHelper(img))

    return images

def saveImages(images, gestureName, targetDir):
    targetDirFullPath = os.path.join(os.getcwd(), targetDir)
    imageId = 0
    for img in images:
        imageId += 1
        cv2.imwrite(os.path.join(targetDirFullPath, gestureName+'_'+str(imageId)+'.jpg'), img)



def getImageList(imageSourceFolder):
    onlyfiles = [ f for f in os.listdir(imageSourceFolder) if os.path.isfile(os.path.join(imageSourceFolder,f)) ]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread( os.path.join(imageSourceFolder, onlyfiles[n]) )
    return images

def getLabel(filename):

    if 'rock' in filename:
        label = ROCK
    elif 'paper' in filename:
        label = PAPER
    else:
        label = SCISSOR
    return label


def getImageLabels(imageFolder):
    onlyfiles = [ f for f in os.listdir(imageFolder) if os.path.isfile(os.path.join(imageFolder,f)) ]
    y_true = np.empty(len(onlyfiles), dtype=int)
    for n in range(0, len(onlyfiles)):
        y_true[n] = getLabel(onlyfiles[n])
    return y_true


def extractHOG(image, hogDescriptorPath):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L1', visualize=True)
    cv2.imshow('img', hog_image)
    cv2.waitKey(0)



# import the necessary packages
import argparse
import cv2
import numpy as np
import os
from skimage.feature import hog

ROCK=1
PAPER=2
SCISSOR=3
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imageDir", required=True, help="Path to input image directory")
args = vars(ap.parse_args())

srcImages = getImageList(args["imageDir"])
for image in srcImages:
    transformedImage = transformImage(image, 300)
    images = augmentImage(transformedImage)
    saveImages(images, 'rock', 'img_dest')

imageLabels = getImageLabels('img_dest')

#ToDo
# 1. Apply HOG, a higher level feature instead of using raw pixel values
# 2. Apply PCA to reduce feture dimentions!, that will speed up training/testing!

im = cv2.imread('img_dest/rock_1.jpg')
extractHOG(im, 'config/hog')

import imgaug as ia
from imgaug import augmenters as iaa
from os import listdir, mkdir
from os.path import isfile, join, isdir
import imageio
import shutil
import cv2
import numpy as np

# Transformation functions
rotations = [
    [iaa.Affine(
        rotate=(-30, 30),
        cval=255
    )],
    [iaa.Affine(
        rotate=(-60, 60),
        cval=255
    )],
    [iaa.Affine(
        rotate=(-90, 90),
        cval=255
    )],
    [iaa.Affine(
        rotate=(-120, 120),
        cval=255
    )],
    [iaa.Affine(
        rotate=(-150, 150),
        cval=255
    )],
    [iaa.Affine(
        rotate=(-15, 15),
        cval=255
    )],
    [iaa.Affine(
        rotate=(-45, 45),
        cval=255
    )],
    [iaa.Affine(
        rotate=(-75, 75),
        cval=255
    )],
    [iaa.Affine(
        rotate=(-105, 105),
        cval=255
    )],
    [iaa.Affine(
        rotate=(-135, 135),
        cval=255
    )],
    [iaa.Affine(
        rotate=(-165, 165),
        cval=255
    )],
    [iaa.Affine(
        rotate=(-180, 180),
        cval=255
    )]
]

shears = [
    [iaa.Affine(
        shear=(-10, 10),
        cval=255
    )],
    [iaa.Affine(
        shear=(-20, 20),
        cval=255
    )],
    [iaa.Affine(
        shear=(-30, 30),
        cval=255
    )],
    [iaa.Affine(
        shear=(-40, 40),
        cval=255
    )],
    [iaa.Affine(
        shear=(-50, 50),
        cval=255
    )],
    # [iaa.Affine(
    #     shear=(-60, 60),
    #     cval=255
    # )],
    # [iaa.Affine(
    #     shear=(-48, 48),
    #     cval=255
    # )],
    # [iaa.Affine(
    #     shear=(-54, 54),
    #     cval=255
    # )],
    # [iaa.Affine(
    #     shear=(-60, 60),
    # )],
    # [iaa.Affine(
    #     shear=(-66, 66),
    # )],
    # [iaa.Affine(
    #     shear=(-72, 72),
    # )],
    # [iaa.Affine(
    #     shear=(-78, 78),
    # )]
]

flip = [iaa.Fliplr(1)]

crop = [iaa.Crop(percent=(0, 0.2))]

scale = [iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        cval=255
    )]

# otherTransformations = [
#     iaa.Crop(percent=(0, 0.2)), 
    # iaa.GaussianBlur(sigma=(0, 0.5)),
    # iaa.ContrastNormalization((0.75, 1.5)),
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # iaa.Multiply((0.8, 1.2), per_channel=0.2),
#     iaa.Affine(
#         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#     ),
# ]

# Read images from the inputDir as black and white
def readImages(gesDir):
    print("Reading input images from {}".format(gesDir))
    allImages = {}

    images = []
    for f in listdir(gesDir):
        filepath = join(gesDir, f)
        if isfile(filepath) and '.DS_Store' not in filepath:
            print("Reading {}".format(filepath))
            # Load in grayscale
            image = imageio.imread(filepath, pilmode="RGB")
            # Convert to black and white
            # bw[bw < 128] = 0
            # bw[bw >= 128] = 255
            dim = (200, 200)
            image = cv2.resize(image, dim)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            images.append(res)

    return images

# Write images after transformations
def writeImages(augImages, gesDir):
    print("Writing images to {}".format(gesDir))
    count = 0
    for image in augImages:
        imageio.imwrite(join(gesDir, '{}.jpg'.format(count)), image)
        count += 1
    return count
    

# Perform transformations on the image based on the functions above
def transformImages(images):
    print("Transforming images")     
    ia.seed(1)

    augImages = []
    #Rotate the images
    for rotation in rotations:
        seq = iaa.Sequential(rotation, random_order=True)
        augImages.extend(seq.augment_images(images))

    # shear images
    for shear in shears:
        seq = iaa.Sequential(shear, random_order=True)
        augImages.extend(seq.augment_images(augImages))

    # flip images
    seq = iaa.Sequential(flip, random_order=True)
    augImages.extend(seq.augment_images(augImages))

    # Crop images
    seq = iaa.Sequential(crop, random_order=True)
    augImages.extend(seq.augment_images(augImages))

    # scale images
    seq = iaa.Sequential(scale, random_order=True)
    augImages.extend(seq.augment_images(augImages))
    return augImages

# main
def main():
    inputDir = "./inputImages"
    outputDir = "./augmentedImages"
    gestures = ["Rock", "Paper", "Scissors"]

    # remove outputDir if it exists
    if isdir(outputDir):
        shutil.rmtree(outputDir)
    mkdir(outputDir)
    # shutil.copytree(inputDir, outputDir) # to save original images as well

    # process for each gesture
    count = 0
    for gesture in gestures:
        igesDir = join(inputDir, gesture)
        ogesDir = join(outputDir, gesture)
        mkdir(ogesDir)

        # read, transform and write
        images = readImages(igesDir)
        augImages = transformImages(images)
        count += writeImages(augImages, ogesDir)
        
    print("Total images created : {}".format(count))

main()
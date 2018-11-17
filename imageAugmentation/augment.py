import imgaug as ia
from imgaug import augmenters as iaa
from os import listdir
from os.path import isfile, join, isdir
import imageio
import shutil
import numpy as np

inputDir = "./inputImages"
outputDir = "./augmentedImages"
gestures = ["Rock", "Paper", "Scissors"]

rotations = [
    [iaa.Affine(
        rotate=(0, 0),
    )],
    [iaa.Affine(
        rotate=(-30, 30),
    )],
    [iaa.Affine(
        rotate=(-60, 60),
    )],
    [iaa.Affine(
        rotate=(-90, 90),
    )],
    [iaa.Affine(
        rotate=(-120, 120),
    )],
    [iaa.Affine(
        rotate=(-150, 150),
    )],
    [iaa.Affine(
        rotate=(-15, 15),
    )],
    [iaa.Affine(
        rotate=(-45, 45),
    )],
    [iaa.Affine(
        rotate=(-75, 75),
    )],
    [iaa.Affine(
        rotate=(-105, 105),
    )],
    [iaa.Affine(
        rotate=(-135, 135),
    )],
    [iaa.Affine(
        rotate=(-165, 165),
    )],
    [iaa.Affine(
        rotate=(-180, 180),
    )]
]

flip = [iaa.Fliplr(1)]
    # iaa.Fliplr(1),
    # iaa.Crop(percent=(0, 0.2)), 
    # iaa.GaussianBlur(sigma=(0, 0.5)),
    # iaa.Sometimes(0.5,
        
    # ),
    # iaa.ContrastNormalization((0.75, 1.5)),
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # iaa.Affine(
    #     # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #     # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #     rotate=(-45, 45),
    #     # shear=(-8, 8)
    # ),

def readImages():
    print("Reading input images")
    allImages = {}

    for gesture in gestures:
        Dir = join(inputDir, gesture)
        images = []
        for f in listdir(Dir):
            filepath = join(Dir, f)
            if isfile(filepath) and '.DS_Store' not in filepath:
                print("Reading {}".format(filepath))
                # Load in grayscale
                bw = imageio.imread(filepath, pilmode="L")
                # Convert to black and white
                bw[bw < 128] = 0
                bw[bw >= 128] = 255 
                images.append(bw)
        allImages[gesture] = images

    return allImages

def writeImages(augImages):
    print("Writing images")
    if isdir(outputDir):
        shutil.rmtree(outputDir)
    shutil.copytree(inputDir, outputDir)
    total = 0
    for gesture in gestures:
        count = 0
        for image in augImages[gesture]:
            imageio.imwrite(join(outputDir, gesture, '{}.jpg'.format(count)), image)
            count += 1
        total += count
    print("Created {} augmented images".format(total))


def transformImages(images):
    print("Transforming images")     
    ia.seed(1)

    augImages = {}
    for gesture in gestures:
        augImages[gesture] = []
        # Rotate the images
        for rotation in rotations:
            seq = iaa.Sequential(rotation, random_order=True)
            augImages[gesture].extend(seq.augment_images(images[gesture]))

        # save original images
        # flip images
        seq = iaa.Sequential(flip, random_order=True)
        augImages[gesture].extend(seq.augment_images(augImages[gesture]))
    return augImages


def main():
    images = readImages()
    augImages = transformImages(images)
    writeImages(augImages)


main()
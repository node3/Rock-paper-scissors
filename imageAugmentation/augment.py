import imgaug as ia
from imgaug import augmenters as iaa
from os import listdir
from os.path import isfile, join
import imageio
import shutil

inputDir = "./inputImages"
outputDir = "./augmentedImages"
gestures = ["Rock", "Paper", "Scissors"]

def readImages():
    print("Reading input images")
    allImages = {}

    for gesture in gestures:
        Dir = join(inputDir, gesture)
        images = []
        for f in listdir(Dir):
            filepath = join(Dir, f)
            if isfile(filepath):
                images.append(imageio.imread(filepath, pilmode="L"))
        allImages[gesture] = images

    return allImages

def writeImages(augImages):
    print("Writing images")
    if os.path.isdir(outputDir):
        os.rmdir(outputDir)
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

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        # iaa.Crop(percent=(0, 0.1)), 
        iaa.Crop(px=(0, 16)),
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        iaa.ContrastNormalization((0.75, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)

    augImages = {}
    for gesture in gestures:
        greScale = []
        for images in images[gesture]:
            bw = np.asarray(gray).copy()
        augImages[gesture] = seq.augment_images(images[gesture])
    return augImages

def main():
    images = readImages()
    augImages = transformImages(images)
    writeImages(augImages)


main()
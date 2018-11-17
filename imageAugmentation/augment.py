import imgaug as ia
from imgaug import augmenters as iaa
from os import listdir
from os.path import isfile, join
import imageio

def readImages(directory, size=None, extract=None):
    images = []
    for f in listdir(directory):
        filepath = join(directory, f)
        if isfile(filepath):
            images.append(imageio.imread(filepath, pilmode="RGB"))
    return images

def writeImages(images):
    count = 0
    for image in images:
        imageio.imwrite('{}.jpg'.format(count), image)
        count += 1

def transformImages(images):
    ia.seed(1)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Crop(percent=(0, 0.1)), 
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

    return seq.augment_images(images)

def main():
    images = readImages("./inputImages")
    augImages = transformImages(images)
    writeImages(augImages)


main()
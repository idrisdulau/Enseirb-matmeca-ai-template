import cv2

path = "5.tif"

def cv2_he_ycrcb(img):
    imgYCRCB = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    luma, redChroma, blueChroma = cv2.split(imgYCRCB)

    equalizedLuma = cv2.equalizeHist(luma)

    cv2.merge((equalizedLuma, redChroma, blueChroma), imgYCRCB)
    imgRGB = cv2.cvtColor(imgYCRCB, cv2.COLOR_YCR_CB2BGR)
    return imgRGB

# ------------------------ #

import numpy 

# Fonctions utiles
# img.flatten() | numpy.bincount( _ , minlength=256) | numpy.cumsum()
def normalizedCumulativeHistogram(img):
    pass
    # return cumSumArray

def lookupTable(cumulativeHist):
    return numpy.floor(255 * cumulativeHist).astype(numpy.uint8)

# Fonctions utiles
# img.flatten() | numpy.reshape() | numpy.asarray()
def transform(img, colorMap):
    pass
    # return equalizedImgArray

def he_gray(img):
    sourceImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cumulativeHistArray = normalizedCumulativeHistogram(sourceImage)
    colorMapUint8 = lookupTable(cumulativeHistArray)
    equalizedImgArray = transform(sourceImage, colorMapUint8)

    return equalizedImgArray

def he_ycrcb(img):
    sourceImage = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    luma, redChroma, blueChroma = cv2.split(sourceImage)

    cumulativeHistArray = normalizedCumulativeHistogram(luma)
    colorMapUint8 = lookupTable(cumulativeHistArray)
    equalizedLuma = transform(luma, colorMapUint8)

    equalizedImgArray = cv2.merge((equalizedLuma, redChroma, blueChroma))
    imgArrayRGB = cv2.cvtColor(equalizedImgArray, cv2.COLOR_YCR_CB2BGR)

    return imgArrayRGB


# Un exemple avec cv2.equalizeHist :
img = cv2.imread(path)
cv2.imshow('Source Image',img)
img2 = cv2_he_ycrcb(img)
cv2.imshow('YCRCB Transform',img2)
cv2.waitKey(0)


# Votre egalisation d'histogramme manuelle
# img = cv2.imread(path)

# cv2.imshow('Source Image', img)
# cv2.imshow('Grayscale Transform', he_gray(img))
# cv2.imshow('YCRCB Transform', he_ycrcb(img))
# cv2.waitKey(0)

import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage import data
from skimage import color
from skimage import exposure
import numpy as np


def detect_line_canny():
    """
    Wykyrwanie krawddzi przy pomocy algorytmu Canny'egp
    """
    img = cv2.imread('2.jpeg')
    cv2.imshow('image', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 100, 200)
    cv2.imshow('line', edges)

    key = cv2.waitKey(0)


def detector_descriptor_ORB():
    """
    Deskryptor ORB

    """

    img1 = cv2.imread('box.png',0)
    img2 = cv2.imread('box2.png',0)

    # initiate detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    keyp1, des1 = orb.detectAndCompute(img1, None)
    keyp2, des2 = orb.detectAndCompute(img2, None)

    #create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #Match descriptors
    matches = bf.match( des1,des2)

    #Sort them in the order of theri distance
    matches = sorted(matches, key= lambda x: x.distance)

    #Draw first 20 matchers
    img3 = cv2.drawMatches(img1, keyp1, img2, keyp2, matches[:20], flags=2, outImg=None)

    plt.imshow(img3)
    plt.show()


def detector_descriptor_SIFT():
    """
    Deskryptor SIFT
    """
    img1 = cv2.imread('box.png',0)
    img2 = cv2.imread('box2.png',0)

    # initiate detector
    orb = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with ORB
    keyp1, des1 = orb.detectAndCompute(img1, None)
    keyp2, des2 = orb.detectAndCompute(img2, None)

    #create BFMatcher object
    bf = cv2.BFMatcher()

    #Match descriptors
    matches = bf.match( des1,des2)

    #Sort them in the order of theri distance
    matches = sorted(matches, key= lambda x: x.distance)

    #Draw first 20 matchers
    img3 = cv2.drawMatches(img1, keyp1, img2, keyp2, matches[:20], flags=2, outImg=None)

    plt.imshow(img3)
    plt.show()


def descriptor_LBP():
    """
    Deskryptor wysokopoziomowy LBP
    """
    img = cv2.imread('2.jpeg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius = 1
    n_points = 4 * radius

    lbp = local_binary_pattern(img_gray, n_points, radius, 'default')
    cv2.imshow('new', lbp)
    key = cv2.waitKey(0)


def descriptor_HoG():
    """
    Deskryptor Histograms of OrientedGradients
    """

    img = color.rgb2gray(data.astronaut())

    _, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)


    fig, (ax1, ax2) = plt.subplots(1, 2)
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    fig.suptitle('Descriptor HoG')

    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Women Input')

    #Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_adjustable('box-forced')
    ax2.set_title('Historam of Oriented Gradzients')

    plt.show()


def Hougha_transform():

    img = cv2.imread('3.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Input', gray)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_line_length = 100
    max_line_gap = 10

    lines = cv2.HoughLinesP(edges, 1,np.pi/180, 100, min_line_length, max_line_gap )
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow('Output',img)
    cv2.waitKey(0)


if __name__ == '__main__':
    Hougha_transform()

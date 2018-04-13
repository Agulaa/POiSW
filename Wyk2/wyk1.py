import cv2
import matplotlib.pyplot as plt
import numpy as np


def hist():

    img = cv2.imread("put.png")
    cv2.imshow("image", img)
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    plt.plot(hist)
    plt.xlim(0,255)
    plt.show()
    key = cv2.waitKey(0)


def new_hist():
    """
    Wyrównywanie histogramu
    """
    img = cv2.imread("put.png")
    cv2.imshow('img', img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(img)
    cv2.imshow('equalized img', equalized)


    hist = cv2.calcHist([equalized], [0], None, [256], [0,256])
    plt.plot(hist)
    plt.ylim(0,256)
    plt.show()
    key = cv2.waitKey(0)


def color_hist():

    img = cv2.imread("2.jpeg",)
    cv2.imshow("image", img)
    color = ('r', 'g', 'b')
    for channel, c in enumerate(color):
        hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
        plt.subplot(1,3, channel+1)
        plt.title('Channel {}, color {} '.format(channel, c))
        plt.plot(hist, color = c)
        plt.xlim([0, 256])

    plt.show()
    key = cv2.waitKey(0)


def new_img():
    """"
    Poprawa jakosci->  nowy_obraz  = alfa * stary_obraz + beta

    """
    alfa = 1.5
    beta = -40.0
    img = cv2.imread("1.jpg")
    cv2.imshow("Old image", img)

    img = img.astype('int32')
    img_new = alfa * img + beta
    img_new = np.clip(img_new, 0, 255)
    img_new = img_new.astype('uint8')

    cv2.imshow("New img", img_new)
    key = cv2.waitKey(0)


def threshold_image():
    img = cv2.imread('put.png', 0)
    cv2.imshow("img", img)

    _, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    adapt_thresh_1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    adapt_thresh_2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, adapt_thresh_3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('normal threshold', threshold)
    cv2.imshow('adaptive thresh mean', adapt_thresh_1)
    cv2.imshow('adaptive thresh gaussian', adapt_thresh_2)
    cv2.imshow('normal and otsu', adapt_thresh_3)



    key = cv2.waitKey(0)

def scale_image():
    img = cv2.imread('1.jpg')
    cv2.imshow('img', img)

    img_scale_nearset = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    img_scale_linear = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    img_scale_cubic = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    img_scale_lanczos4 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)

    cv2.imshow('Nearset scaled image ', img_scale_nearset)
    cv2.imshow('Linear scaled image ', img_scale_linear)
    cv2.imshow('Cubic scaled image ', img_scale_cubic)
    cv2.imshow('Lanczos4 scaled image ', img_scale_lanczos4)

    key = cv2.waitKey(0)


def change_perspective_():
    img = cv2.imread('3.jpg')

    src = np.float32([[68, 82], [490, 62], [30, 530], [509, 530]])
    dst = np.float32([[0, 0], [650, 0], [0, 650], [650, 650]])
    T = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, T, (650, 650))

    plt.subplot(1,2,1)
    img = cv2.rectangle(img, (50, 82), (509, 530), (30, 200, 0), 5)
    plt.imshow(img)
    plt.title('Input')
    plt.subplot(1,2,2)
    plt.imshow(warped)
    plt.title('Output')
    plt.show()

if __name__ == '__main__':
    change_perspective_()
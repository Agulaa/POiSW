import cv2
import numpy as np
from matplotlib import pyplot as plt

def trackbar_color():
    """
    Show in real time color with use 3 parameters R,G,B
    :return:
    """
    def nothing(x):

        pass

    # Create a black image, a window
    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R', 'image', 0, 255, nothing)
    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('B', 'image', 0, 255, nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch, 'image')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]

    cv2.destroyAllWindows()

def zad1():

    def nothing(x):
        print("Trackbar reporting for duty with value: " + str(x))
        pass

        # Create a black image, a window

    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R', 'image', 0, 255, nothing)
    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('B', 'image', 0, 255, nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch, 'image')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]

    cv2.destroyAllWindows()

def zad2_1():
    img = cv2.imread('flaming.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()

    cv2.waitKey(0)

def zad2_2():
    def nothing_(x):
        pass
    img = cv2.imread('put.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('image')


    cv2.createTrackbar("Type", 'image',0,5,nothing_)
    cv2.createTrackbar("Number 1", 'image', 0,255, nothing_)
    cv2.createTrackbar("Number 2", 'image', 0, 255, nothing_)


    while(1):

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


        i_1 = cv2.getTrackbarPos('Number 1', 'image')
        i_2 =  cv2.getTrackbarPos('Number 2', 'image')
        t = cv2.getTrackbarPos("Type", 'image')
        if t == 0:
            img2 = img

        if t == 1:
            ret, thresh1 = cv2.threshold(img, i_1, i_2, cv2.THRESH_BINARY)
            img2 = thresh1
        if t == 2:
            ret, thresh2 = cv2.threshold(img, i_1, i_2,cv2.THRESH_BINARY_INV)
            img2 = thresh2
        if t == 3:
            ret, thresh3 = cv2.threshold(img, i_1, i_2, cv2.THRESH_TRUNC)
            img2= thresh3
        if t == 4:
            ret, thresh4 = cv2.threshold(img, i_1, i_2,cv2.THRESH_TOZERO)

            img2 = thresh4
        if t == 5:
            ret, thresh5 = cv2.threshold(img, i_1, i_2,cv2.THRESH_TOZERO_INV)
            img2= thresh5
        cv2.imshow('image', img2)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad3():
    img_to_scale = cv2.imread('qr.jpg')
    cv2.imshow('original',img_to_scale )

    resize = cv2.resize(img_to_scale, (0,0), fx=1.175, fy=1.75)
    resize_img_linear = cv2.resize(img_to_scale, (0,0), fx=1.175, fy=1.75, interpolation=cv2.INTER_LINEAR)
    resize_img_nearest = cv2.resize(img_to_scale, (0,0), fx=1.175, fy=1.75, interpolation=cv2.INTER_NEAREST)
    resize_img_area = cv2.resize(img_to_scale, (0, 0), fx=1.175, fy=1.75, interpolation=cv2.INTER_AREA)
    resize_img_lanczos4 = cv2.resize(img_to_scale, (0, 0), fx=1.175, fy=1.75, interpolation=cv2.INTER_LANCZOS4)

    cv2.imread('normal', resize)
    cv2.imshow('linear', resize_img_linear)
    cv2.imshow('nearest', resize_img_nearest)
    cv2.imshow('area', resize_img_area)
    cv2.imshow('lanczos4', resize_img_lanczos4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad4_1():
    img = cv2.imread('qr.jpg')
    img1 = cv2.resize(img, dsize =(313, 313), interpolation=cv2.INTER_LANCZOS4)
    img2 = cv2.imread('logo.png')

    dst = cv2.addWeighted(img1, 0.7, img2, 0.3,0)

    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad4_2():
    def nothing1(x):
        pass

    img = cv2.imread('qr.jpg')
    img1 = cv2.resize(img, dsize=(313, 313), interpolation=cv2.INTER_LANCZOS4)
    img2 = cv2.imread('logo.png')

    cv2.namedWindow('image')

    cv2.createTrackbar("alpha", 'image', 0, 10, nothing1)
    cv2.createTrackbar("betha", 'image', 0, 10, nothing1)

    while (1):

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        a = cv2.getTrackbarPos("alpha", 'image')
        b = cv2.getTrackbarPos("betha", 'image')
        dst = cv2.addWeighted(img1, a/10, img2, b/10, 0)
        cv2.imshow('image', dst)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad5():
    img = cv2.imread('qr.jpg')

    e1 = cv2.getTickCount()
    img_scale = cv2.resize(img, (0, 0), fx=1.75, fy=1.75)
    e12 = cv2.getTickCount()

    e2 = cv2.getTickCount()
    linear = cv2.resize(img, (0, 0), fx=1.75, fy=1.75, interpolation=cv2.INTER_LINEAR)
    e22 = cv2.getTickCount()

    e3 = cv2.getTickCount()
    nearest = cv2.resize(img, (0, 0), fx=1.75, fy=1.75, interpolation=cv2.INTER_NEAREST)
    e32 = cv2.getTickCount()

    e4 = cv2.getTickCount()
    area = cv2.resize(img, (0, 0), fx=1.75, fy=1.75, interpolation=cv2.INTER_AREA)
    e42 = cv2.getTickCount()

    e5 = cv2.getTickCount()
    lanczos4 = cv2.resize(img, (0, 0), fx=1.75, fy=1.75, interpolation=cv2.INTER_LANCZOS4)
    e52 = cv2.getTickCount()

    time1 = (e12 - e1) / cv2.getTickFrequency()
    time2 = (e22 - e2) / cv2.getTickFrequency()
    time3 = (e32 - e3) / cv2.getTickFrequency()
    time4 = (e42 - e4) / cv2.getTickFrequency()
    time5 = (e52 - e5) / cv2.getTickFrequency()
    print("Time scale: ", (time1 * 1000), " miliseconds")
    print("Time linear: ", (time2 * 1000), " miliseconds")
    print("Time nearest: ", (time3 * 1000), " miliseconds")
    print("Time area: ", (time4 * 1000), " miliseconds")
    print("Time lanczos4: ", (time5 * 1000), " miliseconds")





if __name__ == '__main__':
    zad5()

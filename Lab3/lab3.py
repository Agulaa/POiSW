import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

def zad1_1():
    """
    2D Convolution - Image Filtering
    :return:
    """
    img1 = cv2.imread('lena_noise.jpg')
    img2 = cv2.imread('lena_salt_and_pepper.jpg')

    kernel1 = np.ones((5, 5), np.float32) / 25
    dst1 = cv2.filter2D(img1, -1, kernel1)

    kernel2 = np.ones((5, 5), np.float32) / 25
    dst2 = cv2.filter2D(img1, -1, kernel2)

    plt.subplot(221)
    plt.imshow(img1)
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(222)
    plt.imshow(dst1)
    plt.title('Averaging')
    plt.xticks([]), plt.yticks([])

    plt.subplot(223)
    plt.imshow(img2)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(224)
    plt.imshow(dst2)
    plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad1_12():
    """
    2D Convolution (image filtering ) with variable filtering window
    :return:
    """
    def nothing(x):
        pass

    img1 = cv2.imread('lena_noise.jpg')
    cv2.namedWindow('image')
    cv2.createTrackbar('X', 'image', 1, 10, nothing)
    dst1 = img1
    while(1):
        cv2.imshow('image', dst1)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        y = cv2.getTrackbarPos('X', 'image')

        kernel1 = np.ones((y, y), np.float32) / y**2
        dst1 = cv2.filter2D(img1, -1, kernel1)

        cv2.imshow('image', dst1)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad1_2():
    """
    Image blurring
    :return:
    """
    img1 = cv2.imread('lena_noise.jpg')
    img2 = cv2.imread('lena_salt_and_pepper.jpg')

    blur1 = cv2.blur(img1, (5, 5))
    blur2 = cv2.blur(img2, (5, 5))

    plt.subplot(221)
    plt.imshow(img1)
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(222)
    plt.imshow(blur1)
    plt.title('Blurred')
    plt.xticks([]), plt.yticks([])

    plt.subplot(223)
    plt.imshow(img2)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(224)
    plt.imshow(blur2)
    plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad1_22():
    """
    Image blurring with variable filtering window
    :return:
    """
    def nothing(x):
        pass
    img1 = cv2.imread('lena_noise.jpg')

    cv2.namedWindow('image')
    cv2.createTrackbar('X', 'image', 1, 10, nothing)
    blur1 = img1
    while(1):
        cv2.imshow('image', blur1)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        y = cv2.getTrackbarPos('X', 'image')

        blur1 = cv2.blur(img1, (y, y))


        cv2.imshow('image', blur1)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad1_3():
    """
    Gaussian Filtering
    :return:
    """

    img1 = cv2.imread('lena_noise.jpg')
    img2 = cv2.imread('lena_salt_and_pepper.jpg')

    blur1 = cv2.GaussianBlur(img1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(img2, (5, 5), 0)

    plt.subplot(221)
    plt.imshow(img1)
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(222)
    plt.imshow(blur1)
    plt.title('GaussianBlur')
    plt.xticks([]), plt.yticks([])

    plt.subplot(223)
    plt.imshow(img2)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(224)
    plt.imshow(blur2)
    plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad1_4():
    """
    Median Filtering
    :return:
    """
    img1 = cv2.imread('lena_noise.jpg')
    img2 = cv2.imread('lena_salt_and_pepper.jpg')

    media1 = cv2.medianBlur(img1, 5)
    median2 = cv2.medianBlur(img2, 5)

    plt.subplot(221)
    plt.imshow(img1)
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(222)
    plt.imshow(media1)
    plt.title('Median Blur')
    plt.xticks([]), plt.yticks([])

    plt.subplot(223)
    plt.imshow(img2)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(224)
    plt.imshow(median2)
    plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad1_5():
    """
    Bilateral Filtering
    :return:
    """
    img1 = cv2.imread('lena_noise.jpg')
    img2 = cv2.imread('lena_salt_and_pepper.jpg')

    blur1 = cv2.bilateralFilter(img1, 9, 75, 75)
    blur2 = cv2.bilateralFilter(img2, 9, 75, 75)

    plt.subplot(221)
    plt.imshow(img1)
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(222)
    plt.imshow( blur1)
    plt.title('Bilateral Filter')
    plt.xticks([]), plt.yticks([])

    plt.subplot(223)
    plt.imshow(img2)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(224)
    plt.imshow(blur2)
    plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad2():
    """
    Morphological Transformations -> erosion and dilation
    :return:
    """
    def nothing(x):
        pass

    img = cv2.imread('j.png',0)

    cv2.namedWindow('erosion')
    _, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.createTrackbar('X', 'erosion', 1, 10, nothing)
    cv2.createTrackbar('Type', 'erosion', 0,1,nothing)

    while(1):
        cv2.imshow('erosion', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        t = cv2.getTrackbarPos('Type', 'erosion')
        y = cv2.getTrackbarPos('X', 'erosion')
        if t == 0:
            kernel = np.ones((y, y), np.uint8)
            erosion = cv2.erode(threshold, kernel, iterations=1)
            img = erosion
        if t == 1:
            kernel = np.ones((y, y), np.uint8)
            dilation = cv2.dilate(threshold, kernel, iterations=1)
            img = dilation

def zad3_1():

    #img1 = mpimg.imread('0.jpg')
    #img2 = mpimg.imread('0.jpg')

    img1 = cv2.imread("pies.jpg")
    img2 = cv2.imread("pies.jpg")

    plt_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    plt_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    hist1 = cv2.calcHist([plt_img1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([plt_img2], [0], None, [256], [0, 256])

    plt.subplot(221)
    plt.imshow(plt_img1)
    plt.title('Color')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(222)
    plt.title('Hist')
    plt.xticks([])
    plt.yticks([])
    plt.plot(hist1)
    plt.xlim(0,255)

    plt.subplot(223)
    plt.imshow(plt_img2, cmap='gray')
    plt.title('Gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(224)
    plt.title('Hist')
    plt.xticks([])
    plt.yticks([])
    plt.plot(hist2)
    plt.xlim(0,255)
    plt.show()


def zad3_2():
    """
    Histograms Equalization
    :return:
    """
    img = cv2.imread("1.jpg",0)

    equ = cv2.equalizeHist(img)
    result = np.hstack((img,equ))
    cv2.imshow('output', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad3_3():
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    :return:
    """
    img = cv2.imread("1.jpg", 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe = clahe.apply(img)
    result = np.hstack((img, clahe))
    cv2.imshow('output', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def zad4():
    """
    draws a circle where we double-click
    :return:
    """
    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img, (x, y), 100, (255, 0, 0), -1)

    # Create a black image, a window and bind the function to window
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while (1):
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

def change_perspective():
    img = cv2.imread('droga.jpg')
    cv2.imshow('1', img )
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    src = np.float32([[540, 466], [687, 466], [0, 763], [1226, 765]]) # lewy górny róg, prawy górny, lewy dolny, prawy górny
    dst = np.float32([[0, 0], [600, 0], [0, 700], [600, 700]])
    T = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, T, (600, 700))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Input')
    plt.subplot(1,2,2)
    plt.imshow(warped)
    plt.title('Output')
    plt.show()


def zad4_1():

    def draw_dot(event,x,y,flags,param):

            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(img,(x,y), 5,(200,25,25),-1)

    img = cv2.imread('droga.jpg')
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_dot)

    while (1):
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
   change_perspective()

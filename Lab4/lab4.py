import cv2
import numpy as np
import pprint
import matplotlib.pyplot as plt
def zad1():
    img = cv2.imread('notbad.PNG', cv2.IMREAD_GRAYSCALE)
    img_re = cv2.resize(img, (1200,900))
    #cv2.imshow('not bad', img_re)

    _, threshold = cv2.threshold(img_re, 50, 255, cv2.THRESH_BINARY)
    #cv2.imshow('thershold', threshold)

    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(threshold, kernel, iterations=1)

    img_cont, contours, hierarchy = cv2.findContours(img_dilation, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_re, contours, -1, (100, 255, 0), 3)
    #cv2.imshow('contours', img_cont)
    cv2.imshow('not bad', img_re)
    perspective = []
    if len(contours) > 4:
        for x in range(1,5):
            cnt = contours[x]
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            perspective.append([cx,cy])

    print(perspective)

    src = np.float32([perspective[3], perspective[2], perspective[1],
                      perspective[0]])  # lewy górny róg, prawy górny, lewy dolny, prawy dolny
    dst = np.float32([[0, 0], [700, 0], [0, 600], [700, 600]])
    T = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_re, T, (700, 600))
    cv2.imshow('change', warped)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad2():
    img_rgb = cv2.imread('mario.PNG')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('mario_coin.PNG', 0)
    #template = cv2.imread('mario_coin.jpg', 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imshow('res.png', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    zad2()
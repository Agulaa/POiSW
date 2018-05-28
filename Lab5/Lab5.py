import cv2
import numpy as np

def zad1():
    img = cv2.imread('pies.jpg', 0)
    cv2.imshow('Input dog', img)

    fast = cv2.FastFeatureDetector_create(threshold=15,nonmaxSuppression= True)
    key_point = fast.detect(img, None)
    print(len(key_point))
    #print(key_point)
    img2 = cv2.drawKeypoints(img, key_point,img,  color=(255,0,0))

    cv2.imshow('Output dog', img2)

    cv2.waitKey()


def zad2():
    img = cv2.imread('pies.jpg', 0)


    # orb detector
    orb = cv2.ORB_create()
    key_points = orb.detect(img, None)
    keypoints, descriptors = orb.compute(img, key_points)


    #translation
    rows, cols = img.shape

    M = np.float32([[1, 0, 100], [0, 1, 50]])
    trans_img = cv2.warpAffine(img, M, (cols, rows))

    key_points_t = orb.detect(trans_img, None)
    keypoints_t, descriptors_t = orb.compute(trans_img, key_points_t)



    #brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors, descriptors_t)
    matches = sorted(matches, key=lambda x: x.distance)
    res = cv2.drawMatches(img, keypoints,trans_img,key_points_t, matches[:10], None, flags=2)

    cv2.imshow('result', res)

    cv2.waitKey()

def zad3():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    img = cv2.imread('klasa.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    zad2()
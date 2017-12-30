import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def faceDetect(img):
    img=cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        img=cv2.resize(img, (55, 55))

        return img

    max_x = 0
    max_y = 0
    max_w = 0
    max_h = 0
    for (x, y, w, h) in faces:
        # 框出所有人脸 取其中最大
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if w >= max_w:
            max_x = x
            max_y = y
            max_w = w
            max_h = h
    # 框出人脸区域的完整输入图像
    # 其中最大的人脸图像
    largestFace = img[max_y:max_y + max_h, max_x:max_x + max_w]
    largestFace=cv2.resize(largestFace,(55,55))


    return largestFace


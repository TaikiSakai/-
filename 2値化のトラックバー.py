import cv2
import numpy as np
import matplotlib.pyplot as plt

from tkinter import filedialog

img = cv2.imread(filedialog.askopenfilename(), 0)





'''
2値化のトラックバー
'''
def th_trackbar(position):
    global threshold
    threshold = position

cv2.namedWindow('img')
threshold = 100
cv2.createTrackbar('track', 'img', threshold, 255, th_trackbar)

while True:
    ret, img_th = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)#黒が強調
    #img_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,#白が強調
    #                               cv2.THRESH_BINARY, 3, threshold)
    cv2.imshow('img', img_th)
    if cv2.waitKey(10) == 27: #esc key
        break

#cv2.destroyAllWindows()
#cv2.waitKey(1)

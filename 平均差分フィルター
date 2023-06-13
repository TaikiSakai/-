
import cv2
import numpy as np

import copy
from tkinter import filedialog


def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

'''
畳み込み平滑フィルター&平均差分フィルター
'''
def convolution(img):
    kernel = np.ones((40, 40)) / 1600
    img_ke1 = cv2.filter2D(img, -1, kernel)#第二引数ピット深度
    #平均差分
    diff_img = cv2.absdiff(img, img_ke1)
    
    return diff_img


def main():
    img = cv2.imread(filedialog.askopenfilename(), 0)
    conv_img = convolution(img)
    show(conv_img)


if __name__ == "__main__":
    main()

import cv2
import numpy as np

import copy
from tkinter import filedialog

def gaussian_blur(img):
    img_gaus =cv2.GaussianBlur(img, (9, 9), 2)#9x9 は奇数であること　第三引数はσ

    return img_gaus

'''
畳み込み(平滑化フィルター)
'''
def img_convolution(img):
    kernel = np.ones((3, 3)) / 9.0 #3x3のすべての要素が1(9で割るので1/9)のフィルター
    img_ke1 = cv2.filter2D(img, -1, kernel)#第二引数ビット深度
    subtracted_img = cv2.absdiff(img, img_ke1)
    return subtracted_img

'''
引き算
absdiff
subtract
どっちか使う
'''



img = cv2.imread(filedialog.askopenfilename())
img2 = img_convolution(img)
cv2.imshow("img", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

'''
https://teratail.com/questions/376568
'''
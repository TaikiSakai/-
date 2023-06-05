import cv2
import numpy as np
import matplotlib.pyplot as plt

from tkinter import filedialog

img = cv2.imread(filedialog.askopenfilename(), 0)


'''
画像を表示する
'''
def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


'''
ヒストグラムの作成
'''
def c_histgram(img): #カラー画像
    color_list = ['blue', 'green', 'red']

    for i, j in enumerate(color_list):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=j)

    plt.show()

#c_histgram(img1)

def g_histgram(img): #グレースケール
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.show()

    return hist

#g_histgram(img1)


'''
ヒストグラム均一化
'''
def eq_hist(img):
    #hist = g_histgram(img)
    img_eq = cv2.equalizeHist(img)
    hist_eq = g_histgram(img_eq)

    return img_eq

#img_eq = eq_hist(img)
#cv2.imshow('img', img_eq)


'''
トラックバー
'''
def onTrackbar(position):
    global trackvalue
    trackvalue = position

#trackvalue = 100
#cv2.namedWindow('img')
#cv2.createTrackbar('track', 'img', trackvalue, 255, onTrackbar)



'''
画像の微分, エッジ検出(sobel, laplacian)
'''
#sobel filter
def sobel(img):
    img_sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)#1, 0はX方向に微分する
    img_sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)  # 1, 0はY方向に微分する

    img_sobelx = cv2.convertScaleAbs(img_sobelx)#微分値を0-255の正の値に変換する
    img_sobely = cv2.convertScaleAbs(img_sobely)

    return img_sobelx, img_sobely

#img_gx, img_gy = sobel(img)
#cv2.imshow('gx', img_gx)
#cv2.imshow('gy', img_gy)


#laplacian filter
def laplacian(img):
    img_lap = cv2.Laplacian(img, cv2.CV_32F)
    img_lap = cv2.convertScaleAbs(img_lap)
    #微分値が弱い場合は二倍する
    img_lap *= 2

    return img_lap

#img_lap = laplacian(img)
#cv2.imshow('gx', img_lap)


def gaussiann_blur(img):
    img_blur = cv2.GaussianBlur(img, [3, 3], 2)
    img_lap2 = laplacian(img_blur)
    #微分値が弱い場合は二倍する
    img_lap2 *= 2

    return img_lap2

#img_glap = gaussiann_blur(img)
#cv2.imshow('gx', img_glap)


'''
畳み込み(平滑化フィルター)
'''
def img_convolution(img):
    kernel = np.ones((3, 3)) / 9.0 #3x3のすべての要素が1(9で割るので1/9)のフィルター
    img_ke1 = cv2.filter2D(img, -1, kernel)#第二引数ビット深度

    return img_ke1

#img_ke1 = cv2.imshow('img', img_ke1)

#img_convolution(img)

'''
画像の微分, エッジ検出(canny法)
'''
def canny(img):
    img_canny = cv2.Canny(img, 10, 100)#img, 閾値1, 閾値2

    return img_canny


#img_canny = canny(img)
#cv2.imshow('img', img_canny)


'''
平滑化
'''
def stabilize(img):#blur
    img_blur = cv2.blur(img, (3, 3))#3x3 filter

    return img_blur

#img_b = stabilize(img)
#cv2.imshow('img', img_b)


def gaussian_blur(img):
    img_gaus =cv2.GaussianBlur(img, (9, 9), 2)#9x9 は奇数であること　第三引数はσ

    return img_gaus

#img_gaus = gaussian_blur(img)
#v2.imshow('img', img_gaus)

#バイラテラルフィルタ　エッジを保存しながら変化の少ない部位を平滑化する
def bilateral(img):
    img_bi = cv2.bilateralFilter(img, 20, 30, 30)

    return img_bi

#img_bi = bilateral(img)
#cv2.imshow('img', img_bi)

'''
############
'''
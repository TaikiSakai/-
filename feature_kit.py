import cv2
import numpy as np
import matplotlib.pyplot as plt

from tkinter import filedialog
import sys
import copy

'''
https://qiita.com/fukuit/items/546f19d2abf98eccd3e7
https://aicam.jp/tech/opencv3/akaze
https://ohke.hateblo.jp/entry/2019/08/03/235500
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_feature2d/py_matcher/py_matcher.html
'''
img0 = filedialog.askopenfilename()
img = cv2.imread(img0)
img_g = cv2.imread(img0, 0)

'''
画像を表示する
'''
def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


'''
特徴点マッチング
'''
def match(from_img, to_img, from_key_points, to_key_points):
    bf_matcher = cv2.BFMatcher_create(cv2.NORM_HAMING, True)
    matches = bf_matcher.match(from_img, to_img)
    mached_img = cv2.drawMatches(
        from_img, from_key_points, to_img, to_key_points,
        matches, None, flags=2
    )

        return mached_img


'''
cornerHARRIS
'''
def haris(img, img_g):
    img_harris = copy.deepcopy(img)
    img_dst = cv2.cornerHarris(img_g, 2, 3, 0.04) #第二引数検出するブロックの大きさ 第三引数ソーベルフィルタ
    #極大値の5%よりも大きい場合は赤でプロットする
    img_harris[img_dst > 0.05 * img_dst.max()] = [0, 0, 255]

    return img_harris

#img_h = haris(img, img_g)
#show(img_h)


'''
SHIFT
'''


'''
ORB
'''
def orb(img):
    img_orb = copy.deepcopy(img)
    orb = cv2.ORB_create()
    kp2 = orb.detect(img_orb)
    img_orb = cv2.drawKeypoints(img_orb, kp2, None)

    return img_orb

#img_or = orb(img)
#show(img_or)


'''
KAZE
'''
def kaze(img):
    img_kaze = copy.deepcopy(img)
    kaze = cv2.KAZE_create()
    kp1 = kaze.detect(img_kaze, None)
    img_kaze = cv2.drawKeypoints(img_kaze, kp1, None)

    return img_kaze

#img_k = kaze(img)
#show(img_k)


'''
AKAZE
'''
def akaze(img):
    img_akaze = copy.deepcopy(img)
    akaze = cv2.AKAZE_create()
    kp1 = akaze.detect(img_akaze, None)
    img_akaze = cv2.drawKeypoints(img_akaze, kp1, None)

    return img_akaze

#img_ak = akaze(img)
#show(img_ak)


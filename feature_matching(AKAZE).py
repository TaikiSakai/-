import cv2
import numpy as np

import copy
from tkinter import filedialog

'''
日本語のファイル名禁止
'''

class FeatureMatching:

    def __init__(self, from_img, to_img):
        self.from_img = from_img
        self.to_img = to_img

        self.from_kimg = None
        self.to_kimg = None
        self.from_kp = None
        self.kp_1 = None
        self.kp = None
        self.desc_1 = None
        self.desc_2 = None

        self.matches = None
        self.matched_img = None
        


    def kaze(self):
        #detetct keypoint with akaze
        self.from_kimg = copy.deepcopy(self.from_img)
        self.to_kimg = copy.deepcopy(self.to_img)
        detector = cv2.KAZE_create(threshold = 1e-3)#閾値を設定する
        self.from_kp = detector.detect(self.from_kimg)
        self.to_kp = detector.detect(self.to_kimg)
        self.from_kimg = cv2.drawKeypoints(self.from_kimg, self.from_kp, None)
        self.to_kimg = cv2.drawKeypoints(self.to_kimg, self.to_kp, None)

        #特徴記述子の取得
        self.kp_1, self.desc_1 = detector.detectAndCompute(self.from_img, None)
        self.kp_2, self.desc_2 = detector.detectAndCompute(self.to_img, None)


    def matching(self):
        matcher = cv2.BFMatcher()
        self.matches = matcher.knnMatch(self.desc_1, self.desc_2, k=2)
        print(self.matches)

        bf = cv2.drawMatchesKnn(self.from_kimg, self.kp_1, self.to_kimg,
                                self.kp_2, self.matches, None)

        return bf


    def select_features(self):
        ratio = 3.5 #0.8
        good = []

        for m, n in self.matches:#m, n はknnの近傍点の数
            if m.distance < ratio * n.distance:
                good.append([m])

        if len(good) <= 1:
            print("cannot detect matching points")

            return None, None, None

        good = sorted(good, key=lambda x: x[0].distance)
        print("valid point numbers", len(good))

        point_num = 20
        if len(good) < point_num:
            point_num = len(good)

        result_img = cv2.drawMatchesKnn(self.from_kimg, self.kp_1, self.to_kimg, 
                                        self.kp_2, good[:point_num], None, flags=0)

        return result_img


#Utility
def cut(img):
    img_c = img[0:957, 0:1280]

    return img_c


def resize(img):
    img_s = cv2.resize(img, dsize=(400, 300))

    return img_s


def main():
    img_1 = cv2.imread(filedialog.askopenfilename(), 0)
    img_2 = cv2.imread(filedialog.askopenfilename(), 0)

    #img_1 = cut(img_1)
    #img_2 = cut(img_2)
    #img_1 = resize(img_1)
    #img_2 = resize(img_2)

    a = FeatureMatching(img_1, img_2)
    a.kaze()

    b = a.matching()
    cv2.imshow('img', b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    c = a.select_features()
    cv2.imshow('img', c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()



        
from tkinter import filedialog
import copy

import cv2
import numpy as np

#特徴点抽出を外で行う

img = cv2.imread(filedialog.askopenfilename())

class FeatureMatching:

    def __init__(self, from_img, to_img, from_key_points, to_key_points,
                 from_description, to_description):
        self.from_img = from_img
        self.to_img = to_img
        self.from_key_points = from_key_points
        self.to_key_points = to_key_points
        self.from_description = from_description
        self.to_description = to_description


    def match(self):
        #↓ attribute error
        bf_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)
        matches = bf_matcher.match(self.from_description, self.to_description)
        matched_img = cv2.drawMatches(
        self.from_img, self.from_key_points, self.to_img, self.to_key_points,
        matches, None, flags=2
    )

        return matched_img


def kaze(img):
    img_kaze = copy.deepcopy(img)
    kaze = cv2.KAZE_create()
    kp, description = kaze.detectAndCompute(img_kaze, None)
    img_kaze = cv2.drawKeypoints(img_kaze, kp, None)

    return img_kaze, kp, description

img_k, key_points, description = kaze(img)

#show(img_k)

a = FeatureMatching(img_k, img_k, key_points, key_points, description, description)
a.match()

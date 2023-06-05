import cv2
import numpy as np
from tkinter import filedialog

class TrackBar:

    def __init__(self, img):
        self.img = img
        self.threshold = 100

    def th_trackbar(self, position):
        self.threshold = position

    def create_window(self):
        cv2.namedWindow('img')
        cv2.createTrackbar('track', 'img', self.threshold, 255, self.th_trackbar)

    def show_window(self):
        while True:
            #ret, img_th = cv2.threshold(self.img, self.threshold, 255,
            #                            cv2.THRESH_BINARY)#黒が強調
            img_th = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,#白が強調
                                           cv2.THRESH_BINARY, 3, self.threshold)
            cv2.imshow('img', img_th)
            if cv2.waitKey(10) == 27:
                break


def create_window():
    img = cv2.imread(filedialog.askopenfilename(), 0)
    if (type(img) is np.ndarray):
        bar = TrackBar(img)
        bar.create_window()
        bar.show_window()

    else:
        print('Image is not read, or not ndarray')

def main():
    create_window()


if __name__ == '__main__':
    main()


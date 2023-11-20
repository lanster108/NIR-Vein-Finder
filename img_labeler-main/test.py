import cv2
import numpy as np

img = cv2.imread('apple.jpg')

def sToneCurve(frame):
    look_up_table = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        look_up_table[i][0] = 255 * (np.sin(np.pi * (i / 255 - 1 / 2)) + 1) / 2
    return cv2.LUT(frame, look_up_table)

image_contrasted = sToneCurve(img)

cv2.imwrite('apple_dark.jpg', image_contrasted)

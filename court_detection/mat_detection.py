"""
+Convert BGR to HSV
+Make the image binary (black and white) using H-value
+Erosion, Dilation to get rid of things not related to mat
-Canny edge detector to detect lines
-Hough circle transformation
"""
import cv2
import numpy as np


def main():
    IMG = '../images/empty_mat.png'
    img = cv2.imread(IMG)
    cv2.imshow('original', img)

    # BGR to HSV===============================================
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', hsv_img)

    # Make image binary========================================
    # Get correct hsv values
    hsv_vals = rgb_to_hsv(255, 128, 74)  # Orange
    # hsv_vals = rgb_to_hsv(39, 72, 123)  # Blue
    h_val = get_h_val(hsv_vals)

    # Create binary image
    bin_img = binary_img_by_h_value(hsv_img, h_val)
    # cv2.imshow('thresh', bin_img)

    # Erosion and Dilation=====================================
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(bin_img, kernel, iterations=4)
    # cv2.imshow('dilation', dilation)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(dilation, kernel, iterations=5)
    # cv2.imshow('erosion', erosion)

    # Canny edge detection====================================
    # Noise reduction. gaussian filter
    blur = cv2.GaussianBlur(erosion, (7, 7), 0)
    # cv2.imshow('blur', blur)

    canny = cv2.Canny(blur, 100, 200)
    cv2.imshow('Canny', canny)

    # Hough circles===========================================
    # detect circles in the image

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def binary_img_by_h_value(img, h1):
    low_val = np.array([h1-10, 100, 100])
    high_val = np.array([h1+10, 255, 255])
    thresh = cv2.inRange(img, low_val, high_val)
    return thresh


def rgb_to_hsv(r, g, b):
    rgb_mat = np.uint8([[[r, g, b]]])
    hsv = cv2.cvtColor(rgb_mat, cv2.COLOR_RGB2HSV)
    # print(hsv)
    return hsv


def get_h_val(hsv_values):
    # print(hsv_values[0][0][0])
    return hsv_values[0][0][0]


main()


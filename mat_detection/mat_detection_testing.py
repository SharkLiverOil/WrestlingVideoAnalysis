import cv2
import numpy as np
from matplotlib import pyplot as plt

IMG = '../images/mat1.png'


def main():
    img = cv2.imread(IMG)
    cv2.imshow('Original', img)

    hsv = bgr_to_hsv(img)
    cv2.imshow('HSV', hsv)
    cv2.imwrite('../images/hsv_mat1.png', hsv)

    orange = rgb_to_hsv([[[255, 98, 0]]])
    pink = rgb_to_hsv(([[[255, 0, 157]]]))
    red = rgb_to_hsv([[[255, 0, 0]]])
    blue = rgb_to_hsv([[[0, 34, 255]]])

    thresh_col = threshold_img_by_color(hsv)
    cv2.imshow('color thresh', thresh_col)
    ret, inv_thresh = cv2.threshold(thresh_col, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('color thresh binary inv', inv_thresh)

    erod = erosion(thresh_col)
    cv2.imshow('erosion', erod)

    dilate = dilation(erod)
    cv2.imshow('dilation', dilate)

    edges = canny_edge(dilate)
    cv2.imshow('edges', edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def canny_edge(img):
    #kernel = np.ones((100, 100), np.uint8)
    edges = cv2.Canny(img, 100, 200)
    return edges


def dilation(img):
    kernel = np.ones((5,5), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=1)
    return dilate


def erosion(img):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=3)
    return erosion


def bgr_to_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv


def threshold_img_by_color(hsv_img):
    # Define range of color
    low = np.array([106, 100, 100])  # blue range
    high = np.array([126, 255, 255])

    thresh = cv2.inRange(hsv_img, low, high)
    return thresh


def thresholding(img):
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    cv2.imshow(titles[1], thresh1)
    cv2.imshow(titles[2], thresh2)
    cv2.imshow(titles[3], thresh3)
    cv2.imshow(titles[4], thresh4)
    cv2.imshow(titles[5], thresh5)


def rgb_to_hsv(color_matrix):
    color_matrix = np.uint8(color_matrix)
    hsv = cv2.cvtColor(color_matrix, cv2.COLOR_RGB2HSV)
    print(hsv)
    return hsv


main()
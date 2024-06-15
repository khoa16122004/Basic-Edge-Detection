from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import math
from simple_edge_detector import *


def get_indx_neighbors(i, j, oriented):
    if math.isclose(oriented, -math.pi):
        return i, j + 1
    elif oriented < -math.pi / 2:
        return i - 1, j + 1
    elif math.isclose(oriented, -math.pi / 2):
        return i - 1, j
    elif oriented < 0:
        return i, j - 1
    elif math.isclose(oriented, math.pi / 2):
        return i + 1, j - 1
    elif oriented < math.pi / 2:
        return i + 1, j
    elif math.isclose(oriented, math.pi):
        return i + 1, j + 1
    else:
        return i, j + 1

def canny_edge_detect(img_path, filter_name="sobel", threshold=150):
    img = cv.imread(img_path)
    img = cv.GaussianBlur(img,(3,3),0)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    x_img, y_img, magnimtude_img, oriented_img = edge_operator(img, filter_name)
    
    w, h = img_gray.shape
    Gx, Gy, kernel_size = get_kernel_size(filter_name)
    final_image = np.zeros((w, h))
    
    for i in range(w - kernel_size + 1):
        for j in range(h - kernel_size + 1):
            i_neibor, j_neibor  = get_indx_neighbors(i, j, oriented_img[i, j])
            if magnimtude_img[i, j] > magnimtude_img[i_neibor, j_neibor]:
                print("Truth")
                final_image[i, j] = magnimtude_img[i,j]
    
    edge_img = np.where(final_image > threshold, 255, 0).astype(np.uint8)

    cv.imwrite(f"result/mag_cany.png", magnimtude_img)
    cv.imwrite(f"result/edge_cany.png", edge_img)
    

img_path = "img\lena.jpg"
canny_edge_detect(img_path)

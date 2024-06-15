from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def get_kernel_size(filter_name="sobel"):
    
    # first order
    
    if filter_name == "sobel":
        
        Gx = np.array([[1.0, 0.0, -1.0], 
                        [2.0, 0.0, -2.0], 
                        [1.0, 0.0, -1.0]])

        Gy = np.array([[1.0, 2.0, 1.0], 
                        [0.0, 0.0, 0.0], 
                        [-1.0, -2.0, -1.0]])
        
        return Gx, Gy, 3
    
    elif filter_name == "sobel_5x5":
            Gx = np.array([
                [-2, -3, 0, 3, 2],
                [-4, -6, 0, 6, 4],
                [-5, -8, 0, 8, 5],
                [-4, -6, 0, 6, 4],
                [-2, -3, 0, 3, 2]
            ], dtype=np.float32)
            
            Gy = np.array([
                [-2, -4, -5, -4, -2],
                [-3, -6, -8, -6, -3],
                [0, 0, 0, 0, 0],
                [3, 6, 8, 6, 3],
                [2, 4, 5, 4, 2]
            ], dtype=np.float32)
            return Gx, Gy, 5
    
    elif filter_name == "prewitt":
        Gx = np.array([[1.0, 0.0, -1.0], 
                        [1.0, 0.0, -1.0], 
                        [1.0, 0.0, -1.0]])

        Gy = np.array([[1.0, 1.0, 1.0], 
                        [0.0, 0.0, 0.0], 
                        [-1.0, -1.0, -1.0]])
        return Gx, Gy, 3
    
    
def gradient_orientation(gx, gy):
    return np.arctan2(gy, gx)

def edge_operator(img, filter_name="sobel"):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    w, h = img_gray.shape
    Gx, Gy, kernel_size = get_kernel_size(filter_name)
    
    x_img, y_img, magnimtude_img, oriented_img = np.zeros((w, h)), np.zeros((w, h)), np.zeros((w, h)), np.zeros((w, h))
    
    
    for i in range(w - kernel_size + 1):
        for j in range(h - kernel_size + 1):
            gx = np.sum(np.multiply(Gx, img_gray[i:i + kernel_size, j:j + kernel_size])) /  2 * kernel_size  # x direction
            gy = np.sum(np.multiply(Gy, img_gray[i:i + kernel_size, j:j + kernel_size])) /  2 * kernel_size  # y direction
            mag = np.sqrt(gx ** 2 + gy ** 2) # magnitude
            oriented = gradient_orientation(gx, gy)
            
            
            x_img[i + 1, j + 1], y_img[i + 1, j + 1] = gx, gy
            magnimtude_img[i + 1, j + 1] =  mag
            oriented_img[i + 1, j + 1] = oriented
                            
    cv.imwrite(f"gray.png", img_gray)
    cv.imwrite(f"result/gx_{filter_name}.png", x_img)
    cv.imwrite(f"result/gy_{filter_name}.png", y_img)
    cv.imwrite(f"result/mag_{filter_name}.png", magnimtude_img)
    # plt.imsave(f"result/mag_{filter_name}.png", magnimtude_img, cmap=plt.get_cmap('gray'))

    return x_img, y_img, magnimtude_img, oriented_img



def edge_detection(img_path, filter_name="sobel", threshold=150):
    img = cv.imread(img_path)
    img = cv.GaussianBlur(img,(5,5),0)
    x_img, y_img, magnimtude_img, oriented_img = edge_operator(img, filter_name)
    edge_img = np.where(magnimtude_img > threshold, 255, 0).astype(np.uint8)
    cv.imwrite(f"result/edge_{filter_name}.png", edge_img)



img_path = "img\lena.jpg"
edge_detection(img_path)
# edge_detection(img_path, "sobel_5x5")
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv






def get_kernel_size(filter_name="sobel"):
    if filter_name == "sobel":
        
        Gx = np.array([[1.0, 0.0, -1.0], 
                        [2.0, 0.0, -2.0], 
                        [1.0, 0.0, -1.0]])

        Gy = np.array([[1.0, 2.0, 1.0], 
                        [0.0, 0.0, 0.0], 
                        [-1.0, -2.0, -1.0]])
        
        return Gx, Gy, 3
    
    elif filter_name == "prewitt":
        Gx = np.array([[1.0, 0.0, -1.0], 
                        [1.0, 0.0, -1.0], 
                        [1.0, 0.0, -1.0]])

        Gy = np.array([[1.0, 1.0, 1.0], 
                        [0.0, 0.0, 0.0], 
                        [-1.0, -1.0, -1.0]])
        return Gx, Gy, 3
    
    
def plot_hist(data):
    plt.hist(data, bins=10, edgecolor='black')  # Chọn số lượng bins và màu viền
    plt.xlabel('Value')
    plt.ylabel('Amount')
    plt.title('Itensity histogram')
    plt.grid(True)
    plt.show()

def edge_operator(img, filter_name="sobel"):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    w, h = img_gray.shape
    Gx, Gy, kernel_size = get_kernel_size(filter_name)
    mag_list = []
    
    x_img, y_img, magnimtude_img = np.zeros((w, h)), np.zeros((w, h)), np.zeros((w, h))
    
    for i in range(w - kernel_size + 1):
        for j in range(h - kernel_size + 1):
            gx = np.sum(np.multiply(Gx, img[i:i + kernel_size, j:j + kernel_size])) /  kernel_size  # x direction
            gy = np.sum(np.multiply(Gy, img[i:i + kernel_size, j:j + kernel_size])) /  kernel_size  # y direction
            mag = np.sqrt(gx ** 2 + gy ** 2) # magnitude
            print(gx)
            print(gy)
            x_img[i + 1, j + 1], y_img[i + 1, j + 1] = gx, gy
            magnimtude_img[i + 1, j + 1] =  mag
            mag_list.append(mag)
                
    cv.imwrite(f"result/gx_{filter_name}.png", x_img)
    cv.imwrite(f"result/gy_{filter_name}.png", y_img)
    cv.imwrite(f"result/mag_{filter_name}.png", magnimtude_img)
    # plt.imsave(f"result/mag_{filter_name}.png", magnimtude_img, cmap=plt.get_cmap('gray'))

    return magnimtude_img



def edge_detection(img_path, filter_name="sobel", threshold=400):
    img = cv.imread(img_path)
    magnimtude_img = edge_operator(img, filter_name) 
    edge_img = np.where(magnimtude_img > threshold, 255, 0).astype(np.uint8)
    cv.imwrite(f"result/edge_{filter_name}.png", edge_img)



img_path = "img\lena.jpg"
edge_detection(img_path)
edge_detection(img_path, "prewitt")
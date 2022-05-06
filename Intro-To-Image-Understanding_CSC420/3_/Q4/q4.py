import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def get_eigenvalues(img_path, save_path, k=3, s=0.7):
    img = cv.imread(img_path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    Ix = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=5)
    Iy = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=5)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    IxIy = np.multiply(Ix, Iy)
    Ix2_blurred = cv.GaussianBlur(Ix2, (k, k), s)
    Iy2_blurred = cv.GaussianBlur(Iy2, (k, k), s)
    IxIy_blurred = cv.GaussianBlur(IxIy, (k, k), s)

    M = np.stack([Ix2_blurred.flatten()] + [IxIy_blurred.flatten()] + [IxIy_blurred.flatten()] + [Iy2_blurred.flatten()], axis=1)

    e1_list = []
    e2_list = []
    for i in (range(M.shape[0])):
        m_i = np.reshape(M[i, :], (2, 2))
        (e1, e2), x = np.linalg.eig(m_i)
        e1_list.append(e1)
        e2_list.append(e2)
    e1, e2 = np.array(e1_list), np.array(e2_list)

    plt.clf()
    plt.scatter(e1, e2)
    plt.xlabel("lambda_1")
    plt.ylabel("lambda_2")
    plt.savefig(save_path + "_Eigen_Value_Plot")
    return e1, e2, img

def detect_corner(img_path, save_path, k=3, s=0.7, thres_e1 = 0.9e6, thres_e2 = 0.9e6):
    e1, e2, img = get_eigenvalues(img_path, save_path, k=k, s=s)
    m, n, x = img.shape
    mask = np.bitwise_and(e1 > thres_e1, e2 > thres_e2).reshape(m, n)
    img[mask] = [255, 0, 0]
    plt.clf()
    plt.imshow(img)
    plt.savefig(save_path)


detect_corner('img1.jpeg', 'img1_corners', thres_e1 = 0.9e6, thres_e2 = 0.9e6, s = 0.7)
    detect_corner('img2.jpeg', 'img2_corners', thres_e1 = 0.2e7, thres_e2 = 0.2e7, s = 0.7)

# Different window function sigma:
detect_corner('img1.jpeg', 'img1_corners_with_s=10', thres_e1 = 0.9e6, thres_e2 = 0.9e6, s=10)
detect_corner('img2.jpeg', 'img2_corners_with_s=10', thres_e1 = 0.2e7, thres_e2 = 0.2e7, s = 10)



import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def crop_image(img, tao):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    m, n = img.shape
    m = (m//tao)*tao
    n = (n//tao)*tao
    cropped_img = img[:m, :n]
    return cropped_img

def gradient(img, threshold, tao):
    img = crop_image(img, tao)
    gradx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    grady = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
    gradient_mag = np.sqrt(gradx ** 2 + grady ** 2)
    gradient_mag = np.where(gradient_mag < threshold, 0, gradient_mag)
    directions = np.arctan2(grady, gradx)
    directions = np.where(directions < - np.pi / 12., directions + np.pi, directions)
    directions = np.where(directions > 11 * np.pi / 12., directions - np.pi, directions)

    img_cropped = img
    return gradient_mag, directions, img_cropped

def get_histogram(img, threshold, tao):
    gradient_mag, directions, img_cropped = gradient(img, threshold, tao)
    m, n = gradient_mag.shape
    small_m = int(m//tao)
    small_n = int(n//tao)
    h_mag = np.zeros((small_m, small_n, 6))
    h_count = np.zeros((small_m, small_n, 6))

    for i in range(m):
        for j in range(n):
            row = int(i // tao)
            column = int(j // tao)
            bin = int((directions[i, j] + np.pi / 12.)//(np.pi / 6.))
            h_mag[row, column, bin] += gradient_mag[i, j]
            h_count[row, column, bin] += 1
    return h_mag, h_count, img_cropped


def quiver_plot(img, threshold, tao, img_num):
    h_mag, h_count, img_cropped = get_histogram(img, threshold, tao)
    small_m, small_n, x = h_mag.shape
    h_mag_threshold = np.where(h_mag <= 30000, h_mag, 30000)
    meshX, meshY = np.meshgrid(np.linspace(tao / 2, (small_n - 1) * tao + tao / 2, small_n),
                               np.linspace(tao / 2, (small_m - 1) * tao + tao / 2, small_m))
    #plot magnitude:
    figure1, axis1 = plt.subplots(figsize = (20, 20))
    axis1.imshow(img_cropped, cmap = 'gray')

    for n in range(6):
        axis1.quiver(meshX, meshY,
                    np.sin(n * np.pi / 6) * h_mag_threshold[:, :, n],
                    np.cos(n * np.pi / 6) * h_mag_threshold[:, :, n],
                    color='red', linewidth=0.5, headlength=0, headwidth = 1, headaxislength = 0, pivot ='middle')
    figure1.savefig(img_num + '_magnitude', bbox_inches = 'tight', pad_inches = 0)
    #plot count:
    figure2, axis2 = plt.subplots(figsize = (20, 20))
    axis2.imshow(img_cropped, cmap = 'gray')

    for n in range(6):
        axis2.quiver(meshX, meshY,
                     np.sin(n * np.pi / 6) * h_count[:, :, n],
                     np.cos(n * np.pi / 6) * h_count[:, :, n],
                     color = 'red', linewidth = 0.5, headlength = 0, headwidth = 1, headaxislength = 0, pivot ='middle')
    figure2.savefig(img_num + '_count', bbox_inches = 'tight', pad_inches = 0)
    return h_mag, h_count


def normalize(h_mag, E, save_name):
    m_small, n_small, x = h_mag.shape
    normalized_h = []
    for i in range(m_small - 1):
        for j in range(n_small - 1):
            d = np.array([h_mag[i, j], h_mag[i + 1, j], h_mag[i, j + 1], h_mag[i + 1, j + 1]])
            d = d.flatten()
            normalized = d / np.sqrt(np.sum(d ** 2) + E ** 2)
            normalized_h.append(normalized)
    normalized_h = np.array(normalized_h).reshape((m_small - 1), (n_small - 1), 24)
    np.savetxt(save_name, normalized_h.reshape(-1, 24), delimiter=',')
    return normalized_h



tao = 8
image1 = cv.imread("1.jpg")
h1_mag, h1_count = quiver_plot(image1, 500, 8, '1_Quiver')
image2 = cv.imread("2.jpg")
h2_mag, h2_count = quiver_plot(image2, 500, 8, '2_Quiver')
image3 = cv.imread("3.jpg")
h3_mag, h3_count = quiver_plot(image3, 500, 8, '3_Quiver')
normalize(h1_mag, 0.001, '1.txt')
normalize(h2_mag, 0.001, '2.txt')
normalize(h3_mag, 0.001, '3.txt')

image4 = cv.imread("4_with_flash.jpg")
h4_mag, h4_count = quiver_plot(image4, 1000, 20, '4_with_flash_Quiver')
normalized_h4 = normalize(h4_mag, 0.001, '4_with_flash.txt')

image5 = cv.imread("5_without_flash.jpg")
h5_mag, h5_count = quiver_plot(image5, 1000, 20, '5_without_flash_Quiver')
normalized_h5 = normalize(h5_mag, 0.001, '5_without_flash.txt')

# plot the histograms for cell [0, 1] before and after normalizing
plt.clf()
fig = plt.figure()
plt.bar([0, 1, 2, 3, 4, 5], h4_mag[0, 1])
plt.xlabel("bin")
plt.ylabel("magnitude")
plt.savefig('before_normalization_hist')


plt.clf()
fig = plt.figure()
plt.bar(np.arange(0, 24), normalized_h4[0, 1])
plt.ylabel("magnitude")
fig.tight_layout()
plt.savefig('after_normalization_hist')

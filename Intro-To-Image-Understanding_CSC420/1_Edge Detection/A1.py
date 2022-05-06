import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from random import randrange


IMAGE_1_PATH = 'images/image1.jpg'
IMAGE_2_PATH = 'images/image2.jpg'
IMAGE_3_PATH = 'images/image3.jpg'


# Step 1

def Gaussian_Matrix (size, sigma):
    m = []
    sum = 0
    for u in range(-size, size + 1):
        m.append([])
        for v in range(- size, size + 1):
            x = (math.exp(-(u**2+v**2)/(2*sigma**2)))/(2*math.pi*sigma**2)
            m[-1].append(x)
            sum += x
    for u in range(len(m)):
        for v in range(len(m)):
            m[u][v] = m[u][v]/sum
    return m
    


#Step 2

def Gradient_Magnitude(image):
    gray = image
    num_rows, num_columns = gray.shape
    if num_rows < 3 or num_columns < 3:
        print ('Image too small.')
        return
    else:
        Sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
  
        g_x = np.zeros((num_rows - 2, num_columns - 2))
        g_y = np.zeros((num_rows - 2, num_columns - 2))
        for i in range(1, num_rows - 1):
            for j in range(1, num_columns - 1):
                sum_x = 0
                sum_y = 0
                for u in range(-1, 2):
                    for v in range(-1, 2):
                        sum_x += Sobel_x[u + 1, v + 1] * gray[i - u, j - v]
                        sum_y += Sobel_y[u + 1, v + 1] * gray[i - u, j - v]
                g_x[i - 1, j - 1] = sum_x
                g_y[i - 1, j - 1] = sum_y
        g_dist = np.sqrt(g_x**2 + g_y**2)
        return g_dist
    
#Step 3

def Threshold(g_dist):
    sum = 0
    num_rows, num_columns = g_dist.shape
    for u in range(num_rows):
        for v in range(num_columns):
            sum += g_dist[u, v]
            sum = sum/(num_rows * num_columns)
    t_prev = sum
    t_i = sum
    i = 0
    new = np.copy(g_dist)
    flat = new.flatten()
    while i == 0 or abs(t_i - t_prev) > 0.01:
        lower_sum = 0
        upper_sum = 0
        lower_count = 0
        upper_count = 0
        for value in flat:
            if value < t_i:
                lower_sum += value
                lower_count += 1
            else:
                upper_sum += value
                upper_count += 1
        lower_avg = lower_sum/lower_count
        upper_avg = upper_sum/upper_count
        t_prev = t_i
        t_i = (lower_avg + upper_avg)/2
        i += 1
    t = t_i
    for u in range(num_rows):
        for v in range(num_columns):
            if new[u, v] > t:
                new[u, v] = 255
            else:
                new[u, v] = 0
    return new
    

#Step 4 TEST  

#Step 1 
plt.clf()
matrix1 = Gaussian_Matrix(3, 3)
plt.imshow(matrix1)
plt.colorbar()
plt.savefig('matrix_1.jpg')

plt.clf()
matrix2 = Gaussian_Matrix(3, 8)
plt.imshow(matrix2)
plt.colorbar()
plt.savefig('matrix_2.jpg')


#Step 2
img1 = cv2.imread(IMAGE_1_PATH) 
image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread(IMAGE_2_PATH) 
image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img3 = cv2.imread(IMAGE_3_PATH) 
image3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

plt.clf()
GM1 = Gradient_Magnitude(image1)
plt.imshow(GM1, cmap = 'gray')
plt.savefig('image1_gradient_magnitude.jpg')

plt.clf()
GM2 = Gradient_Magnitude(image2)
plt.imshow(GM2, cmap = 'gray')
plt.savefig('image2_gradient_magnitude.jpg')

plt.clf()
GM3 = Gradient_Magnitude(image3)
plt.imshow(GM3, cmap = 'gray')
plt.savefig('image3_gradient_magnitude.jpg')


#Step 3
plt.clf()
out1 = Threshold(GM1)
plt.imshow(out1, cmap = 'gray')
plt.savefig('out1.jpg')

plt.clf()
out2 = Threshold(GM2)
plt.imshow(out2, cmap = 'gray')
plt.savefig('out2.jpg')

plt.clf()
out3 = Threshold(GM3)
plt.imshow(out3, cmap = 'gray')
plt.savefig('out3.jpg')

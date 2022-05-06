# Q1.3

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

img = np.full((500, 500), 255.)
img[200:301, 200:301] = 0.
plt.imshow(img, cmap = "gray")
plt.savefig('img')

sigmas = []
response = []
for s in range(100):
    LoG = (s ** 2) * ndimage.gaussian_laplace(img, sigma = s)
    max = np.max(np.abs(LoG))
    response.append(max)
    sigmas.append(s)

plt.clf()
plt.plot(sigmas, response)
plt.title('NLoG Response for Various Sigma Values')
plt.xlabel('sigmas')
plt.ylabel('response')
plt.savefig('Q1_part3')

print('optimal sigma = ' + str(sigmas[np.argmax(np.abs(response))]))






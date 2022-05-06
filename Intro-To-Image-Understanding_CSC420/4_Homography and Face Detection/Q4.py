import cv2
import matplotlib.pyplot as plt
import numpy as np


# def click_event(event, x, y, flags, params):
#
#     # checking for left mouse clicks
#     if event == cv2.EVENT_LBUTTONDOWN:
#
#         # displaying the coordinates
#         # on the Shell
#         print(x, ' ', y)
#
#         # displaying the coordinates
#         # on the image window
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(img, str(x) + ',' +
#                     str(y), (x,y), font,
#                     1, (255, 0, 0), 2)
#         cv2.imshow('image', img)
#
#     # checking for right mouse clicks
#     if event==cv2.EVENT_RBUTTONDOWN:
#
#         # displaying the coordinates
#         # on the Shell
#         print(x, ' ', y)
#
#         # displaying the coordinates
#         # on the image window
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         b = img[y, x, 0]
#         g = img[y, x, 1]
#         r = img[y, x, 2]
#         cv2.putText(img, str(b) + ',' +
#                     str(g) + ',' + str(r),
#                     (x,y), font, 1,
#                     (255, 255, 0), 2)
#         cv2.imshow('image', img)
#
# # driver function
# if __name__=="__main__":
#
#     # reading the image
#     img = cv2.imread('hallway3.jpg')
#
#     # displaying the image
#     cv2.imshow('image', img)
#
#     # setting mouse handler for the image
#     # and calling the click_event() function
#     cv2.setMouseCallback('image', click_event)
#
#     # wait for a key to be pressed to exit
#     cv2.waitKey(0)
#
#     # close the window
#     cv2.destroyAllWindows()

# can change this to 'B' or 'C'
CASE = 'C'

HALLWAY1_WALL = [(1064, 23), (1098, 187), (883, 456), (885, 243), (736, 556),
                 (730, 348)]
HALLWAY2_WALL = [(903, 336), (946, 495), (754, 768), (744, 557), (609, 877),
                 (594, 671)]
HALLWAY3_WALL = [(937, 210), (966, 375), (851, 645), (845, 432), (764, 751),
                 (757, 540)]
HALLWAY1_FLOOR = [(719, 554), (655, 554), (501, 747), (538, 653), (599, 501),
                  (790, 623)]
HALLWAY3_FLOOR = [(754, 749), (687, 749), (449, 950), (530, 854), (653, 695),
                  (798, 811)]

HALLWAY1_WALL_X = np.array([i[0] for i in HALLWAY1_WALL])
HALLWAY1_WALL_Y = np.array([i[1] for i in HALLWAY1_WALL])
HALLWAY2_WALL_X = np.array([i[0] for i in HALLWAY2_WALL])
HALLWAY2_WALL_Y = np.array([i[1] for i in HALLWAY2_WALL])
HALLWAY3_WALL_X = np.array([i[0] for i in HALLWAY3_WALL])
HALLWAY3_WALL_Y = np.array([i[1] for i in HALLWAY3_WALL])
HALLWAY1_FLOOR_X = np.array([i[0] for i in HALLWAY1_FLOOR])
HALLWAY1_FLOOR_Y = np.array([i[1] for i in HALLWAY1_FLOOR])
HALLWAY3_FLOOR_X = np.array([i[0] for i in HALLWAY3_FLOOR])
HALLWAY3_FLOOR_Y = np.array([i[1] for i in HALLWAY3_FLOOR])

img1 = cv2.imread('hallway1.jpg')
img2 = cv2.imread('hallway2.jpg')
img3 = cv2.imread('hallway3.jpg')

gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray_3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

if CASE == 'A':
    original_points = HALLWAY1_WALL
    original_img = img1
    original_gray = gray_1
    original_X = HALLWAY1_WALL_X
    original_Y = HALLWAY1_WALL_Y

    after_points = HALLWAY2_WALL
    after_img = img2
    after_gray = gray_2
    after_X = HALLWAY2_WALL_X
    after_Y = HALLWAY2_WALL_Y

elif CASE == 'B':
    original_points = HALLWAY1_WALL
    original_img = img1
    original_gray = gray_1
    original_X = HALLWAY1_WALL_X
    original_Y = HALLWAY1_WALL_Y

    after_points = HALLWAY3_WALL
    after_img = img3
    after_gray = gray_3
    after_X = HALLWAY3_WALL_X
    after_Y = HALLWAY3_WALL_Y

elif CASE == 'C':
    original_points = HALLWAY1_FLOOR
    original_img = img1
    original_gray = gray_1
    original_X = HALLWAY1_FLOOR_X
    original_Y = HALLWAY1_FLOOR_Y

    after_points = HALLWAY3_FLOOR
    after_img = img3
    after_gray = gray_3
    after_X = HALLWAY3_FLOOR_X
    after_Y = HALLWAY3_FLOOR_Y

plt.imshow(original_gray, cmap="gray")
plt.plot(original_X, original_Y, 'rs')
plt.savefig('./out/Q4/case{}/input image points.jpg'.format(CASE))

plt.clf()
plt.imshow(after_gray, cmap="gray")
plt.plot(after_X, after_Y, 'rs')
plt.savefig('./out/Q4/case{}/output image points.jpg'.format(CASE))

A = []
for point1, point2 in zip(original_points, after_points):
    x_i = point1[0]
    y_i = point1[1]
    x_i_a = point2[0]
    y_i_a = point2[1]
    first = [x_i, y_i, 1, 0, 0, 0, -x_i_a * x_i, -x_i_a * y_i, -x_i_a]
    second = [0, 0, 0, x_i, y_i, 1, -y_i_a * x_i, -y_i_a * y_i, -y_i_a]
    A.append(first)
    A.append(second)

A = np.asarray(A)

A = A.T @ A

U, sigma, V = np.linalg.svd(A)
H = V[-1]
H = H.reshape((9, 1))
H = H.reshape(3, 3)
print(H)

def H_transform(img, H):
    new_img = np.zeros((len(img), len(img[0])))
    for y in range(len(img)):
        for x in range(len(img[0])):
            value = img[y, x]
            v = np.array([[x], [y], [1]])
            u = H @ v
            u = np.rint(u / u[2, 0])
            u_1 = int(u[0, 0])
            u_2 = int(u[1, 0])
            if 0 <= u_1 < len(img[0]) and 0 <= u_2 < len(img):
                new_img[u_2, u_1] = value
    return new_img


def H_transform_point(x, y, H):
    v = np.array([[x], [y], [1]])
    u = H @ v
    u = np.rint(u / u[2, 0])
    u_1 = int(u[0, 0])
    u_2 = int(u[1, 0])
    return u_1, u_2

# illustrate transformation H on a square
plt.clf()
square = np.zeros((1000, 1400))
square[390:400, 800:1000] = 155
square[400:600, 1000:1010] = 55
square[400:600, 800:1000] = 255
plt.imshow(square, cmap='gray')
plt.savefig('./out/Q4/case{}/square.jpg'.format(CASE))
square = H_transform(square, H)
plt.clf()
plt.imshow(square, cmap='gray')
plt.savefig('./out/Q4/case{}/square transformed.jpg'.format(CASE))

# Q4.3
plt.clf()
plt.imshow(after_gray, cmap="gray")
plt.plot(after_X, after_Y, 'rs', ms=10)
for x, y in zip(original_X, original_Y):
    u_1, u_2 = H_transform_point(x, y, H)
    plt.plot(u_1, u_2, 'gs')
plt.savefig('./out/Q4/case{}/4.3.jpg'.format(CASE))

# Q4.4
new_img_r = np.zeros((2000, 2500))
new_img_g = np.zeros((2000, 2500))
new_img_b = np.zeros((2000, 2500))

for i in range(2000):
    for j in range(2500):
        if 0 <= i < len(original_gray) and 0 <= j < len(original_gray[0]):
            new_img_r[i + 500, j + 500] = original_gray[i, j]
        x, y = H_transform_point(j - 500, i - 500, H)
        if 0 <= x < len(after_gray[0]) and 0 <= y < len(after_gray):
            value = after_gray[y, x]
            new_img_g[i, j] = value
            new_img_b[i, j] = value


new_img = np.stack((new_img_b, new_img_g, new_img_r), axis=-1)
cv2.imshow('new_img', new_img)
cv2.imwrite('./out/Q4/case{}/new img.jpg'.format(CASE), new_img)



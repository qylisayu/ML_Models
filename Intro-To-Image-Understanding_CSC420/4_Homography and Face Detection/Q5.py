import cv2
import matplotlib.pyplot as plt
import numpy as np

def I_o_U(a, b):
    ra = (a[0], a[1], a[0] + a[2], a[1] + a[3])
    rb = (b[0], b[1], b[0] + b[2], b[1] + b[3])
    dx = min(ra[2], rb[2]) - max(ra[0], rb[0])
    dy = min(ra[3], rb[3]) - max(ra[1], rb[1])
    if (dx >= 0) and (dy >= 0):
        intersection_i = dx * dy
    else:
        intersection_i = 0
    union_i = a[2] * a[3] + b[2] * b[3] - intersection_i
    IoU_i = intersection_i / union_i
    return IoU_i


def get_most_similar_rect(face_boxes_i, prev_rect_detect):
    if len(face_boxes_i) == 1:
        (x_i, y_i, w_i, h_i) = tuple(face_boxes_i[0])
        return (x_i, y_i, w_i, h_i)
    elif len(face_boxes_i) > 1:
        most_similar = face_boxes_i[0]
        best_IoU = I_o_U(face_boxes_i[0], prev_rect_detect)
        for face_box in face_boxes_i:
            IoU = I_o_U(face_box, prev_rect_detect)
            if IoU > best_IoU:
                best_IoU = IoU
                most_similar = face_box
        return most_similar


# 5.1
cap = cv2.VideoCapture('KylianMbappe.mp4')
# capture one frame
ret, frame = cap.read()

# detect a face on the first frame
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_boxes = face_detector.detectMultiScale(frame)

if len(face_boxes) == 0:
    print('no face detected')
    assert (False)

# initialize the tracing window around the (first) detected face
(x, y, w, h) = tuple(face_boxes[0])
track_window = (x, y, w, h)
img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow('mean shift tracking', img)
cv2.imwrite('./out/Q5/first_frame.jpg', img)

#  region of interest for tracking
roi = frame[y:y + h, x:x + w]

# convert the roi to HSV so we can construct a histogram of Hue
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# why do we need this mask? (remember the cone?)
# read the description for Figure 3 in the original Cam Shift paper: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.7673
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                   np.array((180., 255., 255.)))

# form histogram of hue in the roi
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

# normalize the histogram array values so they are in the min=0 to max=255 range
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

plt.clf()
plt.plot(roi_hist)
plt.savefig('./out/Q5/Histogram_1.jpg')

# termination criteria for mean shift: 10 iteration or shift less than 1 pixel
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)




prev_rect_detect = track_window
x_axis = []
IoU_lst = []
i = 2

while True:

    # grab a frame
    ret, frame = cap.read()

    if ret == True:

        # convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # histogram back projection using roi_hist
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # use meanshift to shift the tracking window
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # display tracked window
        x, y, w, h = track_window
        rect_track = (x, y, w, h)
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('mean shift tracking', img)

        face_boxes_i = face_detector.detectMultiScale(frame)
        rect_detect = get_most_similar_rect(face_boxes_i, prev_rect_detect)
        prev_rect_detect = rect_detect

        img = cv2.rectangle(frame, (rect_detect[0], rect_detect[1]), (
        rect_detect[0] + rect_detect[2], rect_detect[1] + rect_detect[3]),
                            (255, 0, 0), 2)
        cv2.imshow('mean shift tracking', img)

        IoU_i = I_o_U(rect_detect, rect_track)
        x_axis.append(i)
        IoU_lst.append(IoU_i)

        if i == 4:
            cv2.imwrite('./out/Q5/5.1min_IoU.jpg', img)
        if i == 81:
            cv2.imwrite('./out/Q5/5.1max_IoU.jpg', img)

        if cv2.waitKey(
                33) & 0xFF == 27:  # wait a bit and exit is ESC is pressed
            break
        i += 1

    else:
        break

plt.clf()
plt.plot(x_axis, IoU_lst)
plt.xlabel('frame')
plt.ylabel('IoU')
plt.savefig("./out/Q5/5.1IoU.jpg")

# get highest IoU frame
IoU_lst = np.asarray(IoU_lst)
max_frame = np.argmax(IoU_lst) + 2
max_IoU = IoU_lst[max_frame - 2]
print('5.1 max IoU = {}, frame = {}'.format(max_IoU, max_frame))

# get lowest IoU frame
min_frame = np.argmin(IoU_lst) + 2
min_IoU = IoU_lst[min_frame - 2]
print('5.1 min IoU = {}, frame = {}'.format(min_IoU, min_frame))

percentage_higher_than_70 = sum(i > 0.7 for i in IoU_lst) / len(IoU_lst)
print('5.1 percentage IoU higher than 70%: {}'.format(percentage_higher_than_70))

percentage_lower_than_60 = sum(i < 0.6 for i in IoU_lst) / len(IoU_lst)
print('5.1 percentage IoU lower than 60%: {}'.format(percentage_lower_than_60))

cv2.destroyAllWindows()
cap.release()


# part 2

cap = cv2.VideoCapture('KylianMbappe.mp4')
# capture one frame
ret, frame = cap.read()

# detect a face on the first frame
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_boxes = face_detector.detectMultiScale(frame)

if len(face_boxes) == 0:
    print('no face detected')
    assert (False)

# initialize the tracing window around the (first) detected face
(x, y, w, h) = tuple(face_boxes[0])
track_window = (x, y, w, h)

#  region of interest for tracking
roi = frame[y:y + h, x:x + w]

# convert the roi to HSV so we can construct a histogram of Hue
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blur_roi = cv2.GaussianBlur(gray_roi, (5, 5), 7)
Ix_roi = cv2.Sobel(blur_roi, cv2.CV_64F, 1, 0, ksize=5)
Iy_roi = cv2.Sobel(blur_roi, cv2.CV_64F, 0, 1, ksize=5)
mag_roi, ang_roi = cv2.cartToPolar(Ix_roi, Iy_roi, angleInDegrees=True);
mask = cv2.inRange(mag_roi, 0.05 * mag_roi.max(), mag_roi.max())
roi_hist = cv2.calcHist([np.uint16(ang_roi)], [0], mask, [24], [0, 360])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX);

plt.clf()
plt.plot(roi_hist)
plt.savefig('./out/Q5/Histogram_2.jpg')


# termination criteria for mean shift: 10 iteration or shift less than 1 pixel
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

prev_rect_detect = track_window
x_axis = []
IoU_lst = []
i = 2

while True:

    # grab a frame
    ret, frame = cap.read()

    if ret == True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 7)
        Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
        mag, ang = cv2.cartToPolar(Ix, Iy, angleInDegrees=True)

        ang[mag < 0.05 * mag.max()] = 0

        dst = cv2.calcBackProject([np.uint16(ang)], [0], roi_hist, [0, 360], 1)

        # use meanshift to shift the tracking window
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # display tracked window
        x, y, w, h = track_window
        rect_track = (x, y, w, h)
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        face_boxes_i = face_detector.detectMultiScale(frame)
        rect_detect = get_most_similar_rect(face_boxes_i, prev_rect_detect)
        prev_rect_detect = rect_detect

        img = cv2.rectangle(frame, (rect_detect[0], rect_detect[1]), (
            rect_detect[0] + rect_detect[2], rect_detect[1] + rect_detect[3]),
                            (255, 0, 0), 2)
        cv2.imshow('mean shift tracking', img)

        IoU_i = I_o_U(rect_detect, rect_track)
        x_axis.append(i)
        IoU_lst.append(IoU_i)

        if i == 49:
            cv2.imwrite('./out/Q5/5.2max_IoU.jpg', img)
        if i == 71:
            cv2.imwrite('./out/Q5/5.2min_IoU.jpg', img)

        if cv2.waitKey(
                33) & 0xFF == 27:  # wait a bit and exit is ESC is pressed
            break
        i += 1

    else:
        break

plt.clf()
plt.plot(x_axis, IoU_lst)
plt.xlabel('frame')
plt.ylabel('IoU')
plt.savefig("./out/Q5/5.2IoU.jpg")

# get highest IoU frame
IoU_lst = np.asarray(IoU_lst)
max_frame = np.argmax(IoU_lst) + 2
max_IoU = IoU_lst[max_frame - 2]
print('5.2 max IoU = {}, frame = {}'.format(max_IoU, max_frame))

# get lowest IoU frame
min_frame = np.argmin(IoU_lst) + 2
min_IoU = IoU_lst[min_frame - 2]
print('5.2 min IoU = {}, frame = {}'.format(min_IoU, min_frame))

percentage_higher_than_80 = sum(i > 0.8 for i in IoU_lst) / len(IoU_lst)
print('5.2 percentage IoU higher than 80%: {}'.format(percentage_higher_than_80))

percentage_lower_than_70 = sum(i < 0.7 for i in IoU_lst) / len(IoU_lst)
print('5.2 percentage IoU lower than 70%: {}'.format(percentage_lower_than_70))

cv2.destroyAllWindows()
cap.release()

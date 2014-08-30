"""
The main script
The script creates a trace of gaze points
"""

import cv2
import numpy as np
import helper

# There are two classifiers in res folder
eye_cascade = cv2.CascadeClassifier("../res/haarcascade_eye_tree_eyeglasses.xml")

vc = cv2.VideoCapture(0)

ret, frame = vc.read()

while ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Values for scaleFactor = 1.2 and for minNeighbors = 1
    eyes = eye_cascade.detectMultiScale(gray, 1.2, 1)

    if len(eyes) != 0:
        eyes = [eyes[0]]

    kernel = np.ones((3, 3), np.uint8)

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # --------------------------------------------
        
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

        x_grad = helper.xgrad(roi_gray)
        y_grad = helper.ygrad(roi_gray)

        out_image = np.zeros((h, w))

        pos = helper.find_center(x_grad, y_grad, out_image)
        
        #out_image = np.array(out_image, dtype = np.uint8)
        #out_image = cv2.cvtColor(out_image, cv2.COLOR_GRAY2BGR)
        
        cv2.circle(roi_color, pos, 5, (0, 0, 255))
        #frame[0 : h, 0 : w] = roi_color

        # --------------------------------------------

    cv2.imshow("preview", frame)

    ret, frame = vc.read()

    # Exits on Esc
    if cv2.waitKey(20) == 27:
        break

cv2.destroyAllWindows()


def find_gaze(x, y, w, h, a, b):
    """
    Calculates the gaze position using the given positions of eye and pupil
    x, y, w, h define the bounding rectangle
    a, b define the center of pupil
    """

    center = [x + 0.5 * w, y + 0.5 * h]

    return [center[0] - a, center[1] - b]

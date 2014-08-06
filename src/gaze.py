"""
The main script
"""

import cv2

vc = cv2.VideoCapture(0)

ret, frame = vc.read()

while ret:
    cv2.imshow("preview", frame)

    ret, frame = vc.read()

    if cv2.waitKey(20) == 27:
        break

cv2.destroyAllWindows()

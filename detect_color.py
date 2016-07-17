import cv2
import numpy as np

cap = cv2.VideoCapture(0)

try:
    while(1):

        # Take each frame
        _, frame = cap.read()

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        sensitivity = 80
        hue_range = 50
        bright_range = 50
        lower_white = np.array([0,0,255-sensitivity])
        upper_white = np.array([255,sensitivity,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Bitwise-AND mask and original image
        # res = cv2.bitwise_and(frame,frame, mask= mask)

        # Search area
        h,w,d = frame.shape
        top = 3 * h / 4
        bot = 3 * h / 4 + 20
        mask[0:top, 0:w] = 0
        mask[bot:h, 0:w] = 0

        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(frame, (cx, cy), 20, (0, 0, 255), -1)
            err = cx - w/2
        
        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        # cv2.imshow('res',res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
except KeyboardInterrupt:
    cv2.destroyAllWindows()

import cv2
import numpy as np

import sys

import rospy
from matplotlib import pyplot as plt

from std_msgs.msg import String
from geometry_msgs.msg import Twist, Vector3


VELOCITY_TOPIC='cmd_vel_mux/input/navi'

pub = rospy.Publisher(VELOCITY_TOPIC, Twist, queue_size=10)
rospy.init_node('navigator', anonymous=False)

def update(lin, ang):
	twist = Twist(Vector3(float(lin), 0.0, 0.0), Vector3(0.0, 0.0, -float(ang) / 1000.0)) 
	pub.publish(twist)

end = False
pause = False


cap = cv2.VideoCapture(0)
fc = cv2.cv.CV_FOURCC(*'MJPG')
video = cv2.VideoWriter('video.avi',fc,20.0,(640,480))

try:
    while(1):

        # Take each frame
        _, frame = cap.read()
        
	# Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        sensitivity = 127
        lower_white = np.array([0,0,255-sensitivity])
        upper_white = np.array([180,95,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_white, upper_white)		

        # Bitwise-AND mask and original image
        # res = cv2.bitwise_and(frame,frame, mask= mask)

        # Search area
        h,w,d = frame.shape
        top = 3 * h / 4
        bot = 3 * h / 4 + 50
        mask[0:top, 0:w] = 0
        mask[bot:h, 0:w] = 0

	# Search line follower
	line_mask = mask
	line_left = 50
	line_right = 150
 	line_mask[0:top, 0:line_left] = 0
	line_mask[0:top, line_right:w] = 0

        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(frame, (cx, cy), 20, (0, 0, 255), -1)
            err = cx - w/2
            update(0.1, err)        
        
        if end == False:
			cv2.imshow('frame',frame)
			cv2.imshow('mask',mask)
			#cv2.imshow('line_mask',hsv)
	video.write(frame)        
	# cv2.imshow('res',res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
			end = True
			break
except KeyboardInterrupt:
	pass    
cv2.destroyAllWindows()
print 'done'
video.release()

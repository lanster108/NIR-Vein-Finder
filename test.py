import cv2
from picamera2 import Picamera2
import numpy as np


lower = np.array([100, 150, 20])
upper = np.array([170, 255, 255]) # (These ranges will detect Yellow)


piCam = Picamera2()
piCam.preview_configuration.main.size=(1280,720)
piCam.preview_configuration.main.format="RGB888"
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()
while True:
        frame=piCam.capture_array()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Converting BGR image to HSV format
        mask = cv2.inRange(img, lower, upper) # Masking the image to find our color
        mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image

        if len(mask_contours) != 0:
            for mask_contour in mask_contours:
                if cv2.contourArea(mask_contour) > 500:
                    x, y, w, h = cv2.boundingRect(mask_contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3) #drawing rectangle
                    
        cv2.imshow("mask image", mask) # Displaying mask image
        cv2.imshow("piCam", frame) # Displaying mask image


        if cv2.waitKey(1)==ord('q'):
             break
        cv2.destroyAllWindows()


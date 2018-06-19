# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
	# show the frame
	
	
	res = cv2.aruco.detectMarkers(gray_image,dictionary)
	if len(res[0]) > 0:
            cv2.aruco.drawDetectedMarkers(gray_image,res[0],res[1])
        
        
 
	# clear the stream in preparation for the next frame
	cv2.imshow("Frame", gray_image)
	key = cv2.waitKey(1) & 0xFF
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

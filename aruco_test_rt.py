import numpy as np
import cv2
from cv2 import aruco
import yaml
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 24
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)

#cap = cv2.VideoCapture(0)
#dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
#dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

markerLength = 0.0525
arucoParams = aruco.DetectorParameters_create()

# Camera calibration matrix and coefficients
with open('calibration.yaml') as f:
    loadeddict = yaml.load(f)
camera_matrix2 = loadeddict.get('camera_matrix')
dist_coeffs2 = loadeddict.get('dist_coeff')
camera_matrix = np.asarray(camera_matrix2)
dist_coeffs = np.asarray(dist_coeffs2)


#print(type(camera_matrix))


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Capture frame-by-frame
    #ret, frame = cap.read()
    image = frame.array
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    #gray = cv2.imread('deneme.jpg',0)
    #res = cv2.aruco.detectMarkers(gray,dictionary)
#   print(res[0],res[1],len(res[2]))

    #if len(res[0]) > 0:
     #   corners, ids, = cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=arucoParams)
    #print(len(corners))
    #print(corners)
    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist_coeffs)
        imgWithAruco = aruco.drawDetectedMarkers(gray, corners, ids) #, (0,255,0))
        if rvec is not None:
            for i in range(len(rvec)):
                imgWithAruco = aruco.drawAxis(imgWithAruco, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.05)
                # Display the resulting frame
                cv2.imshow('frame',imgWithAruco)
    else:
        cv2.imshow('frame',gray)
    
    rawCapture.truncate(0)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()



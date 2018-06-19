import numpy as np
import cv2
from cv2 import aruco
import yaml
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# Get camera  calibration parameters
with open('calibration2.yaml') as f:
    loadeddict = yaml.load(f)
camera_matrix2 = loadeddict.get('camera_matrix')
dist_coeffs2 = loadeddict.get('dist_coeff')
camera_matrix = np.asarray(camera_matrix2)
dist_coeffs = np.asarray(dist_coeffs2)

# Initialize the camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 24
#camera.vflip = True
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    img = frame.array
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h,  w = gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,dist_coeffs,(w,h),1,(w,h))
    #print("roi:" , roi)
        # undistort
    mapx,mapy = cv2.initUndistortRectifyMap(camera_matrix,dist_coeffs,None,newcameramtx,(w,h),5)
    dst = cv2.remap(gray,mapx,mapy,cv2.INTER_LINEAR)
    
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    #print(dst.shape, "/ gray:", gray.shape)
    #print(x,"/",y,"/",w,"/",h)
    
    cv2.imshow('frame',dst)
    
    rawCapture.truncate(0)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
    
from __future__ import print_function

import numpy as np
import cv2
from cv2 import aruco
import yaml # Camera calibration files
# For using the PiCamera
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
# For threading
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
import imutils

# Get camera  calibration parameters
with open('calibration2.yaml') as f:
    loadeddict = yaml.load(f)
camera_matrix2 = loadeddict.get('camera_matrix')
dist_coeffs2 = loadeddict.get('dist_coeff')
camera_matrix = np.asarray(camera_matrix2)
dist_coeffs = np.asarray(dist_coeffs2)

# Aruco parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
markerLength = 0.0525
arucoParams = aruco.DetectorParameters_create()

# Start the video stream in a new thread
vs = PiVideoStream(resolution=(640, 480),framerate=32).start()
time.sleep(2.0)
fps = FPS().start()

while(True):
    
    img = vs.read()
    #img = imutils.resize(img, width=480)
    gray_d = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h,  w = gray_d.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,dist_coeffs,(w,h),1,(w,h))
    
    # undistort
    mapx,mapy = cv2.initUndistortRectifyMap(camera_matrix,dist_coeffs,None,newcameramtx,(w,h),5)
    dst = cv2.remap(gray_d,mapx,mapy,cv2.INTER_LINEAR)
    # crop the image
    x,y,w,h = roi
    gray = dst[y:y+h, x:x+w]
    #cv2.imshow('frame',gray)
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=arucoParams)
    
    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist_coeffs)
        #retval, rvec, tvec	=	cv.aruco.estimatePoseBoard(	corners, ids, board, camera_matrix, dist_coeffs)
        imgWithAruco = aruco.drawDetectedMarkers(gray, corners, ids) #, (0,255,0))
        if rvec is not None:
            objectPoints =np.asarray([[105, 105, 105]], dtype=np.float)
            imagePoints	=	cv2.projectPoints(	objectPoints, rvec[0], tvec[0], camera_matrix, dist_coeffs)
                     
            corner1 = np.append(corners[0][0][0],[0]).transpose().reshape(3,1)
            E0 = np.asarray([[markerLength/2, 0, 0]]).transpose()
            #print('Tvec_0: ',tvec[0])
            #print('****')
            #print('Rvec_0: ',rvec[0])
            #print('****')
            #rotM = cv2.Rodrigues(rvec[0])[0]
            #pos = -np.matrix(rotM).T * np.matrix(tvec[0].T)
            R = cv2.Rodrigues(rvec[0])[0]
            Rt = R.transpose()
            a = Rt.dot(corner1)
            b = a + tvec[0].transpose()
            #pos = -R * np.dot(tvec[0].transpose())
            #print(R)
            #print('****')
            #print(b)
            #print('****')
            
            for i in range(len(rvec)):
                imgWithAruco = aruco.drawAxis(imgWithAruco, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.05)
                # Display the resulting frame
        imgWithAruco = cv2.flip(imgWithAruco, 0)
        cv2.imshow('frame',imgWithAruco)
        
    else:
        gray = cv2.flip(gray, 0)
        cv2.imshow('frame',gray)
               
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    fps.update()
    
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
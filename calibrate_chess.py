import numpy as np
import cv2
from cv2 import aruco, Rodrigues
import yaml
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 24
camera.vflip = True
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

###################################################################################################

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
print (objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

found = 0
no = 0

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
    if ret == True:
        print("Found:", no)
        objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.
        
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners.
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        found += 1
        no = no + 1
        time.sleep(0.2)
            
    #cv2.imshow("img", im_with_keypoints) # display
    #cv2.waitKey(0)
    rawCapture.truncate(0)
    
    if found > 20:
        cv2.imshow("img", img) # display
        cv2.waitKey(0)
        break

# When everything done, release the capture
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Finished finding the parameters")

# It's very important to transform the matrix to list.
data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

with open("calibration_chess.yaml", "w") as f:
    yaml.dump(data, f, default_flow_style=False)

print("Parameters saved")

total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error

print("Total error: ", total_error/len(objpoints))
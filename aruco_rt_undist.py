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


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

markerLength = 0.0525
arucoParams = aruco.DetectorParameters_create()

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    start_time = time.time()

    img = frame.array
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
            print('Tvec_0: ',tvec[0])
            print('****')
            print('Rvec_0: ',rvec[0])
            print('****')
            #rotM = cv2.Rodrigues(rvec[0])[0]
            #pos = -np.matrix(rotM).T * np.matrix(tvec[0].T)
            R = cv2.Rodrigues(rvec[0])[0]
            Rt = R.transpose()
            a = Rt.dot(corner1)
            b = a + tvec[0].transpose()
            #pos = -R * np.dot(tvec[0].transpose())
            print(R)
            print('****')
            print(b)
            print('****')
            
            for i in range(len(rvec)):
                imgWithAruco = aruco.drawAxis(imgWithAruco, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.05)
                # Display the resulting frame
                cv2.imshow('frame',imgWithAruco)
    else:
        cv2.imshow('frame',gray)
       
    rawCapture.truncate(0)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    end_time = time.time()
    print("Total execution time: {}".format(end_time - start_time))
    
cv2.destroyAllWindows()
    
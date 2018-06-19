import numpy as np
import cv2
from cv2 import aruco
import yaml

cap = cv2.VideoCapture(0)
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


while(True):
    # Capture frame-by-frame
    #ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, 0)
    gray = cv2.imread('deneme.jpg',0)
    #res = cv2.aruco.detectMarkers(gray,dictionary)
#   print(res[0],res[1],len(res[2]))

    #if len(res[0]) > 0:
     #   corners, ids, = cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=arucoParams)
    
    
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist_coeffs)
    imgWithAruco = aruco.drawDetectedMarkers(gray, corners, ids) #, (0,255,0))
    for i in range(len(rvec)):
        imgWithAruco = aruco.drawAxis(imgWithAruco, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.05)
            
    # Display the resulting frame
    cv2.imshow('frame',imgWithAruco)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


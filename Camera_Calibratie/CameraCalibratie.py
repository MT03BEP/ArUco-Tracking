"""
Dit script kan worden gebruikt voor het berekenen van de intrinsieke cameramatrix.
Werkt voor zowel een webcam of Basler camera. Kies onderaan het script welke
functies gebruikt moeten worden.
Van https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/ geleend
"""

import cv2
import numpy as np
import glob
import json
from pypylon import pylon

MachineV = False

def MakeCalibrationPicturesWebcam(camera,Ammount=1,path=""):
    
    while Ammount > 0:
        ret,img = camera.read() # grab a frame
        filename = f"{path}\image_{Ammount}.jpg"
        cv2.imwrite(filename, img)
        cv2.imshow("img",img)
        cv2.waitKey(0)
        Ammount -= 1
    cv2.destroyAllWindows()
    
def MakeCalibrationPicturesMV(camera,Ammount=1,path=""):
    
    while Ammount > 0:
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        img = grabResult.Array
        filename = f"{path}\image_MV_{Ammount}.jpg"
        cv2.imwrite(filename, img)
        cv2.imshow("img",img)
        cv2.waitKey(0)
        Ammount -= 1
    cv2.destroyAllWindows()
    
def CalibrateCamera(SchaakbordAfmetingen, images_path):
    # Afmetingen van het schaakbord in kruispunten (min,max)
    CHECKERBOARD = SchaakbordAfmetingen
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 
    
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*8 #mm
    
    # Extracting path of individual image stored in a given directory

    images = glob.glob(f'{images_path}\*.jpg')
    print(len(images))
    if len(images) == 0:
        raise ValueError("Er zijn geen fotos gevonden voor de calibraties")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            imgpoints.append(corners2)
            
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            print(img.shape)
            if img.shape[0] > 1024:
                imgS = cv2.resize(img, (1024,768))
            else: imgS = img
            cv2.imshow('img',imgS)
            cv2.waitKey(1000)
            
    cv2.destroyAllWindows()
 
    
    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    
    return mtx, dist

def SaveCameraCalibration(matrix,distortion,path):
    matrix = matrix.tolist()
    distortion = distortion.tolist()
    data = {"camera_matrix":matrix, "dist_coeff":distortion}
    fname = f"{path}.json"
    with open(fname, "w") as f:
        json.dump(data,f)

# For use in Main_Code
def ReadCameraCalibrationJson(filename):
    with open(filename, 'rb') as f:
        data = json.load(f)
        camera_matrix = np.array(data['camera_matrix'])
        camera_distortion = np.array(data['dist_coeff'])
        print(camera_matrix)
        print(camera_distortion)
    return camera_matrix, camera_distortion




if __name__ == '__main__':
    if MachineV:
        cap = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        cap.StartGrabbing()
        MakeCalibrationPicturesMV(cap,10,"Set1MV")
        cam_matrix, cam_distortion = CalibrateCamera((8,16), "Set1MV")
        SaveCameraCalibration(cam_matrix, cam_distortion,"cam_data_MV")
        ReadCameraCalibrationJson("cam_data_MV.json")
        cap.StopGrabbing()
        cap.Close()

    else:
        camera=cv2.VideoCapture(1)
        #camera_width = 1920
        #camera_height = 1080
        #camera_frame_rate = 30  # 40

        #camera.set(2, camera_width)
        #camera.set(4, camera_height)
        #camera.set(5, camera_frame_rate)
        #MakeCalibrationPicturesWebcam(camera,10)
        cam_matrix, cam_distortion = CalibrateCamera((8,16), "Set2")
        SaveCameraCalibration(cam_matrix, cam_distortion,"cam_data_webcam")
        ReadCameraCalibrationJson("cam_data_webcam.json")
        camera.release()

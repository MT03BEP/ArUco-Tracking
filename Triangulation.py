# Hier is de main code waar alles in gebeurt
# Hier roep je functies aan, functies schrijven kan in losse bestanden! Succes :)
"""
In dit script wordt de marker gevolgd door middel van Triangulation. Dit vereist
2 Basler cameras om te werken. Als er nog geen set calibratiefotos is gemaakt,
moet dit alsnog worden gedaan voor elke verschillende setup. Indien het script
opniew uitgevoerd wordt, kan er voor gekozen worden om de fotos niet te maken.
Zet dan Make_pictures = False. Het script berekent de Intrinsieke en Extrinsieke
camera matrices. Dit duurt even. Vervolgens verschijnen er 2 vensters met de 
camerabeelden. Hierna kan er gestart worden door op "s" te drukken. Het script 
stopt na de ingestelde tijd of door "q" ingedrukt te houden. Dit script is 
ontworpen voor een vergelijking tussen triangulation en de ArUco EstimatePose
functie. De opgenomen data is te vinden onder de map CSV data. 
"""



#________________________________IMPORTS_______________________________________
import cv2
import time
import numpy as np
import keyboard
import glob
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import pandas as pd

from pypylon import pylon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Subfuncties.tijdsonderzoek import Interval

#___________________________INITIALISEER COMPONENTEN_________sq__________________
# Camera instellingen
frame_factor = 1.5      # max 2.5, factor = 1 = 640 x 480 pixel frame
camera_gain = 8.0000

# Maak nieuwe calibratiefotos
Make_pictures = False

# Marker instellingen
marker_size = 100  # Markergrootte
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50) # Gebruikte ArUco dictionary

# Data instellingen
EndT = 60;  # Tijd in seconde dat de run duurt (LET OP: fps*t mag niet veelvoud data_limit zijn!!!)

# Visuele instellingen
Tijdsonderzoek = False; # Wel of niet uitvoeren van een intervalanalyse
Plotting = 0;          # 0 = geen live plot, 1 = live translatieplot, 2 = live rotatieplot
ShowImage = True       # T|F om wel of niet live video weer te geven
Plot_speed = False     # T|F
Plot_acc = False       # T|F

#________________________________FUNCTIES:_____________________________________

def MakeCalibrationPicturesMV(camera1,camera2,Ammount=1,path="",dt=0):
    #Maak calibratiefoto's met 2 MV cameras. 
    #Invoer: Eerste camera, 2e camera, hoeveelheid fotos, locatie, tijd tussen foto's
    mx = Ammount
    while Ammount > 0:
        grabResult1 = camera1.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grabResult2 = camera2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        img1 = grabResult1.Array
        img2 = grabResult2.Array
        
        filename1 = f"{path}\image_MV_1_{Ammount}.jpg"
        filename2 = f"{path}\image_MV_2_{Ammount}.jpg"
        
        cv2.imwrite(filename1, img1)
        cv2.imwrite(filename2, img2)
        
        cv2.imshow("camera 1", img1)
        cv2.imshow("camera 2", img2)
        cv2.waitKey(1)
        time.sleep(dt)
        print(f"Pictures taken ({abs(Ammount-mx)+1}/{mx})")
        Ammount -= 1
    cv2.destroyAllWindows()


def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    #read the synched frames
    images_names = glob.glob(f"{frames_folder}\*.jpg")
    images_names = sorted(images_names)
    c1_images_names = images_names[:len(images_names)//2]
    c2_images_names = images_names[len(images_names)//2:]
   
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv2.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv2.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    rows = 8 #number of checkerboard rows.
    columns = 16 #number of checkerboard columns.
    world_scaling = 15 #change this to the real world square size. Or not.  in mm
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
    
    i = 0
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (8, 16), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (8, 16), None)
        
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            cv2.drawChessboardCorners(frame1, (8,16), corners1, c_ret1)
            cv2.imshow('img', frame1)
 
            cv2.drawChessboardCorners(frame2, (8,16), corners2, c_ret2)
            cv2.imshow('img2', frame2)
            
            cv2.waitKey(500)
            
            print(f"Set done ({i+1}/{len(images_names)//2})")
            
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
        else:
            print(f"Set {i+1} failed, but continuing")
        
        i+=1
        
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
    cv2.destroyAllWindows()
    print(ret)
    return R, T

def CalibrateCamera(SchaakbordAfmetingen, images_path, image_name):
    # Afmetingen van het schaakbord in kruispunten (min,max)
    CHECKERBOARD = SchaakbordAfmetingen
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 
    
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*15 #mm
    
    # Extracting path of individual image stored in a given directory
    images_loc = f'{images_path}\{image_name}_*.jpg'
    images = glob.glob(images_loc)

    if len(images) == 0:
        raise ValueError("Er zijn geen fotos gevonden voor de calibraties")
    
    i=0
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
            
            print(f"Done foto ({i+1}/{len(images)})")
            
            cv2.imshow('img',img)
            cv2.waitKey(500)
        else:
            print(f"Foto {i+1} failed, but continuing")
        i+=1
            
    cv2.destroyAllWindows()
 
    
    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("\nCamera matrix :",mtx)
    print("\ndist : \n",dist)
    
    return mtx, dist

def SaveDataToCSV(path, time, t_x, t_y, t_z):
    # Input: arrays of time,  x, y, z
    data = np.array([time, t_x, t_y, t_z])
    df = pd.DataFrame(data.T, columns=['Time (s)','Translation X','Translation Y','Translation z'])
    df.to_csv(f"{path}.csv", index=False)
    print("Data saved to CSV")

#______________________________SET-UP CAMERA___________________________________
# Kies camerapoort
tlFactory = pylon.TlFactory.GetInstance()
lst_devices = tlFactory.EnumerateDevices()
devices = []

print("Aantal gevonden MV cameras: ",len(lst_devices))
    
if len(lst_devices) != 0:
    cameras = pylon.InstantCameraArray(min(len(lst_devices), 2))
    
    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(lst_devices[i]))
        print("Using device ", cam.GetDeviceInfo().GetModelName())


cap = cameras[0]     
cap.Open()   
cap_2 = cameras[1]
cap_2.Open()

# Camera eigenschappen
camera_width = int(frame_factor*640)
camera_height = int(frame_factor*480)
cap.Width.SetValue(camera_width)
cap.Height.SetValue(camera_height)
cap.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
cap.CenterX.SetValue(True)
cap.CenterY.SetValue(True)
cap.Gain.SetValue(camera_gain)

cap_2.Width.SetValue(camera_width)
cap_2.Height.SetValue(camera_height)
cap_2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
cap_2.CenterX.SetValue(True)
cap_2.CenterY.SetValue(True)
cap_2.Gain.SetValue(camera_gain)

print(f"MV camera width: {camera_width}\nMV camera height: {camera_height}\nMV camera gain: {camera_gain}")

x = np.array([])
y = np.array([])
z = np.array([])
roll_data = np.array([])
pitch_data = np.array([])
yaw_data = np.array([])

#_________________________________MEASURE DATA_________________________________
print("Cameraobject 1: ",cap)
print("Cameraobject 2: ",cap_2)

def cc():
    cap.Close()
    cap_2.Close()

# Maak nieuwe calibratiefotos in nieuwe setup
if Make_pictures:
    MakeCalibrationPicturesMV(cap,cap_2,20,"TriangulationPictures",2)

# Bereken de intrinsieke cameramatrix van beide cameras
mtx1,dist1 = CalibrateCamera((8,16), "TriangulationPictures","image_MV_1")
mtx2,dist2 = CalibrateCamera((8,16), "TriangulationPictures","image_MV_2")

# Voer de stereocalibratie uit
R,t = stereo_calibrate(mtx1, dist1, mtx2, dist2, "TriangulationPictures")

# Bereken de Projectiematrix P1 als primair assenstelsel
RT1 = np.concatenate([np.identity(3), [[0],[0],[0]]], axis = -1)
P1 = mtx1 @ RT1
 
# Bereken de Projectiematrix P2 als secondair assenstelsel met R,t van stereocalibratie
RT2 = np.concatenate([R, t], axis = -1)
P2 = mtx2 @ RT2

x = np.array([])
y = np.array([])
z = np.array([])
x2 = np.array([])
y2 = np.array([])
z2 = np.array([])
T = np.array([])

#_______________________________PREPARATION_LOOP_______________________________
# In deze loop kun je nog aanpassingen aan de camerascherpte doen voor je begint met meten

while True:
    # Pak een frame en vertk tot een grijstinten frame
    grabResult = cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    gray_frame = grabResult.Array
    
    grabResult2 = cap_2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    gray_frame2 = grabResult2.Array 

    # Vind alle hoeken en ids van de makers in het frame
    corners, ids, rejected = aruco.detectMarkers(gray_frame, aruco_dict, mtx1, dist1)
    
    corners2, ids2, rejected = aruco.detectMarkers(gray_frame2, aruco_dict, mtx2, dist2)

    if ShowImage == True:
        cv2.imshow('frame', gray_frame)
        cv2.imshow("frame2", gray_frame2)

    # Stopconditie
    cv2.waitKey(1)

    # Stop met de 'q'-toets
    if keyboard.is_pressed('q'):
        key = ord('q')
        Mainloop = False
        break

    # Start met de 's'-toets
    if keyboard.is_pressed('s'): 
        Mainloop = True
        break
        
#___________________________________MAIN_LOOP__________________________________
# Vanaf hier worden de gemeten gegevens opgeslagen in arrays en csv
if Mainloop:
    start = time.time()
    c = 0
    print("Measurement started")
    while True:
        # Pak een frame en vertk tot een grijstinten frame
        grabResult = cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        gray_frame = grabResult.Array

        grabResult2 = cap_2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        gray_frame2 = grabResult2.Array 
        
        # Record de tijd
        Measured_Time = time.time() - start

        # Vind alle hoeken en markers in frame
        corners,   ids, rejected = aruco.detectMarkers(gray_frame,  aruco_dict, mtx1, dist1)
        corners2, ids2, rejected = aruco.detectMarkers(gray_frame2, aruco_dict, mtx2, dist2)
         
        # Mits in beide frames een markers is gedetecteerd
        if ids is not None and ids2 is not None:
            # Indien nodig kunnen de punten verbetert worden door voor de vervorming te corigeren
            #corners  = cv2.undistortPoints(np.float32(corners[0]),  mtx1, dist1,P=P1)
            #corners2 = cv2.undistortPoints(np.float32(corners2[0]), mtx2, dist2,P=P2)

            # Triangulation
            corners_4D = cv2.triangulatePoints(P1,P2,np.float32(corners[0]),np.float32(corners2[0]))
            corners_3D = cv2.convertPointsFromHomogeneous(corners_4D.T)                       
            
            # Verwerking voor 3D plot
            X_Corn = corners_3D[:,:,0]
            Y_Corn = corners_3D[:,:,1]
            Z_Corn = corners_3D[:,:,2]
            
            # Midden van de marker, opgenomen data
            Midpoint = np.array([np.mean(X_Corn),np.mean(Y_Corn),np.mean(Z_Corn)])
            
            if c == 0:
                X0=X_Corn[0]
                Y0=Y_Corn[0]
                Z0=Z_Corn[0]
            
            # Verwerking voor 3D plot
            vertices = [list(zip(X_Corn.T.tolist()[0],Y_Corn.T.tolist()[0],Z_Corn.T.tolist()[0]))]
            poly = Poly3DCollection(vertices, alpha=0.8)
            
            # Plot het vlak in 3D
            fig = plt.figure(figsize=(30,30))
            ax = fig.add_subplot(111, projection='3d')
            
            rvec_list_all, tvec_list_all, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx1, dist1)
            tvec = tvec_list_all[0][0]
            
            if c == 0:
                tvec0 = tvec
            
            # Assen optioneel
            """
            ax.set_xlim3d(-100000, 100000)
            ax.set_ylim3d(-100000, 100000)
            ax.set_zlim3d(-100000, 100000)
            """
            
            ax.set_title("3D plot")
            ax.set_xlabel('x-axis')
            ax.set_ylabel('y-axis')
            ax.set_zlabel('z-axis')
            
            ax.add_collection3d(poly)
            ax.scatter3D(Midpoint[0],Midpoint[1],Midpoint[2],"red")    # Midden
            ax.scatter3D(X_Corn[0],Y_Corn[0],Z_Corn[0],color="red")    # Hoek 1
            ax.scatter3D(X_Corn[1],Y_Corn[1],Z_Corn[1],color="blue")   # Hoek 2
            ax.scatter3D(X_Corn[2],Y_Corn[2],Z_Corn[2],color="black")  # Hoek 3
            ax.scatter3D(X_Corn[3],Y_Corn[3],Z_Corn[3],color="yellow") # Hoek 4
            ax.scatter3D(0,0,0,color="darkorange")                     # 0-punt assenstelsel
            plt.show()
            
            # Controle
            print(time.time() - start)
            
            # Data record
            x = np.append(x, [Midpoint[0]-X0])
            y = np.append(y, [Midpoint[1]-Y0])
            z = np.append(z, [Midpoint[2]-Z0])
            T = np.append(T, [Measured_Time])

            x2 = np.append(x2, [tvec[0]-tvec0[0]])
            y2 = np.append(y2, [tvec[1]-tvec0[1]])
            z2 = np.append(z2, [tvec[2]-tvec0[2]])
            
            c+=1           
            
        if ShowImage == True:
            cv2.imshow('frame', gray_frame)
            cv2.imshow("frame2", gray_frame2)

        # Stopcondities
        # Stopconditie 1
        if time.time() - start >= EndT:
            SaveDataToCSV("CSV data\Triangulation",T, x, y, z)
            SaveDataToCSV("CSV data\CamNormal",T, x2, y2, z2)
            break
        
        # Stopconditie 2
        if keyboard.is_pressed('q'):
            EndT = time.time() - start
            SaveDataToCSV("CSV data\Triangulation",T, x, y, z)
            SaveDataToCSV("CSV data\CamNormal",T, x2, y2, z2)
            break
        
        cv2.waitKey(1) & 0xFF

#____________________________SYSTEM_WRAP_UP____________________________________    
cap.StopGrabbing()
cap_2.StopGrabbing()

cap.Close()
cap_2.Close()
cv2.destroyAllWindows()

if Tijdsonderzoek and Mainloop:
    Interval(T, EndT)
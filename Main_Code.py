"""
In dit script wordt de marker gevolgd door middel van ArUco estimatePose fuctie. 
Dit vereist 1 of 2 Basler cameras om te werken. Als eerste detecteert het script
een 0 punt voor elke camera. Als er geen 0-punt wordt gedetecteert gaat het 
script niet verder. Vervolgens verschijnen er 1 of 2 vensters met de 
camerabeelden. Hierna kan er gestart worden door op "s" te drukken. Het script 
stopt na de ingestelde tijd of door "q" ingedrukt te houden. De opgenomen data 
is te vinden onder de map CSV data. 
"""
#________________________________IMPORTS_______________________________________
import numpy as np
import cv2
import cv2.aruco as aruco
import math
import time
import matplotlib.pyplot as plt
import keyboard
from pypylon import pylon

# import subfucties
from Camera_Calibratie.CameraCalibratie import ReadCameraCalibrationJson
from Subfuncties.translatie_aardvasteassenstelsel import rotationMatrixToEulerAngles
from Subfuncties.bereken_bewegingen import CalculateDerivative
from Subfuncties.tijdsonderzoek import Interval
from Subfuncties.plotting import Translatieplot, Rotatieplot
from Subfuncties.datatocsv import SaveDataToCSV, ResetDataArrays, ReadCSVToArrays, SaveProcesedDataToCSV

 
#___________________________INITIALISEER COMPONENTEN___________________________
# Camera instellingen
Max_Cameras = 2;      # [1,2], aantal MV cameras in gebruik
frame_factor = 1      # max 2.5

camera_gain = 8.0000

# Marker instellingen
marker_size = 50  # Markergrootte
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50) # Gebruikte ArUco dictionary

# Data instellingen
EndT = 120;  # Tijd in seconde dat de run duurt (LET OP: fps*t mag niet veelvoud data_limit zijn!!!)
data_limit = 4001 # Lente van de data arrays voordat ze worden opgeslagen en geleegd

# Visuele instellingen
Tijdsonderzoek = True; # Wel of niet uitvoeren van een intervalanalyse
Plotting = 0;          # 0 = geen live plot, 1 = live translatieplot, 2 = live rotatieplot
ShowImage = True       # T|F om wel of niet live video weer te geven
Plot_speed = False     # T|F
Plot_acc = False       # T|F


#________________________________FUNCTIES:_____________________________________
# Detecteer het 0-punt met behulp van een marker
def DetectOriginByMarker(camera, aruco_dict, camera_matrix, camera_distortion):
    FoundOrigin = False
    targetLength = 10
    x = np.array([])
    y = np.array([])
    z = np.array([])
    
    while not FoundOrigin:
        # Get frame
        grabResult = camera.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
        gray_frame = grabResult.Array
            
        cv2.imshow('0-punt detectie', gray_frame)
        
        # Find all the corners
        corners, ids, rejected = aruco.detectMarkers(gray_frame, aruco_dict, camera_matrix, camera_distortion)
        
        # Bepaal oriëntatie markers
        rvec_list_all, tvec_list_all, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix,
                                                                                   camera_distortion)
        # Mits marker gevonden
        if ids is not None:
            if len(x)==0:
                print(f"Gevonden marker voor 0-punt is ID: {ids[0]}")
            rvec = rvec_list_all[0][0]
            tvec = tvec_list_all[0][0]

            rotation_matrix, jacobian = cv2.Rodrigues(rvec)

            tx, ty, tz = tvec[0], tvec[1], tvec[2]
            x = np.append(x,tx)
            y = np.append(y,ty)
            z = np.append(z,tz)
            
        if len(x) == targetLength:
            x = sum(x)/targetLength
            y = sum(y)/targetLength
            z = sum(z)/targetLength

            FoundOrigin = True
            print(f"Nulpunt van assenstelsel is {x},{y},{z} tov camera")
            
        cv2.waitKey(1)
    cv2.destroyWindow("0-punt detectie")
    return x, y, z, rotation_matrix, True

#______________________________SET-UP CAMERA___________________________________
# Lees cameracalibratie uit bestand
camera_matrix, camera_distortion = ReadCameraCalibrationJson("Camera_Calibratie\cam_data_MV.json")
# Herschaal intrinsieke matrix
camera_matrix[0,0] = camera_matrix[0,0]* frame_factor
camera_matrix[1,1] = camera_matrix[1,1]* frame_factor

# Verkrijg cameras
tlFactory = pylon.TlFactory.GetInstance()
lst_devices = tlFactory.EnumerateDevices()  
devices = []
    
print("Aantal gevonden MV cameras: ",len(lst_devices))
    
if len(lst_devices) != 0:
    cameras = pylon.InstantCameraArray(min(len(lst_devices), Max_Cameras))
    
    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(lst_devices[i]))
        print("Using device ", cam.GetDeviceInfo().GetModelName())

# Open cameras
cap = cameras[0]
cap.Open()
if Max_Cameras == 2:
    cap_2 = cameras[1]
    cap_2.Open()

# Camera instellingen worden uitgevoerd
camera_width = int(frame_factor*640)
camera_height = int(frame_factor*480)
cap.Width.SetValue(camera_width)
cap.Height.SetValue(camera_height)
cap.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
cap.CenterX.SetValue(True)
cap.CenterY.SetValue(True)
cap.Gain.SetValue(camera_gain)
if Max_Cameras == 2:
    cap_2.Width.SetValue(camera_width)
    cap_2.Height.SetValue(camera_height)
    cap_2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    cap_2.CenterX.SetValue(True)
    cap_2.CenterY.SetValue(True)
    cap_2.Gain.SetValue(camera_gain)

print(f"MV camera width: {camera_width}\nMV camera height: {camera_height}\nMV camera gain: {camera_gain}\n")

# Data Collection initialisatie
T = np.array([])
x = np.array([])
y = np.array([])
z = np.array([])
yaw_data = np.array([])
pitch_data = np.array([])
roll_data = np.array([])

#_________________________________Meet 0-punt__________________________________
print("Cameraobject 1: ",cap)
if Max_Cameras == 2:
    print("Cameraobject 2: ",cap_2)

# Detecteer een eerste 0-punt
x0, y0, z0, rot_mat0, HaveOrigin = DetectOriginByMarker(cap, aruco_dict, camera_matrix, camera_distortion)
Vec_3D0 = np.array([x0, y0, z0])

if Max_Cameras == 2:
    x02, y02, z02, rot_mat02, HaveOrigin2 = DetectOriginByMarker(cap_2, aruco_dict, camera_matrix, camera_distortion)
    Vec_3D02 = np.array([x02, y02, z02])

#_______________________________Voorbereidings_loop____________________________
# In deze loop kun je nog aanpassingen doen voor je begint met meten

while True:
    # Pak een frame en vertk tot een grijstinten frame
    grabResult = cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    gray_frame = grabResult.Array
    if Max_Cameras == 2:
        grabResult2 = cap_2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        gray_frame2 = grabResult2.Array 

    # Vind alle hoeken en ids van de makers in het frame
    corners, ids, rejected = aruco.detectMarkers(gray_frame, aruco_dict, camera_matrix, camera_distortion)
    if Max_Cameras == 2:
        corners2, ids2, rejected = aruco.detectMarkers(gray_frame2, aruco_dict, camera_matrix, camera_distortion)

    #Bereken data cam 1
    if ids is not None:
        # Teken een vierkant om de gedetecteerde marker
        aruco.drawDetectedMarkers(gray_frame, corners, ids)
        # Bepaal de positie van 
        rvec_list_all, tvec_list_all, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix,
                                                                                   camera_distortion)
        rvec = rvec_list_all[0][0]
        tvec = tvec_list_all[0][0]

        cv2.drawFrameAxes(gray_frame, camera_matrix, camera_distortion, rvec, tvec, 100)

        # Bepaal rotatiematrix op basis van rotatievector
        rotation_matrix, jacobian = cv2.Rodrigues(rvec)

        # Maak 3D vector voor translaties
        Vec_3D = tvec - Vec_3D0

        # Roteer de gemeten verplaatsing van V_3D-V_3D0 van camera stelsel naar markerstelsel
        T_Vec_3D = np.dot(Vec_3D, rot_mat0)

        # Bereken de rotatie van marker tov markerstelse
        Rot_Mat_Marker = np.matmul(rot_mat0.T,rotation_matrix)
            
        roll, pitch, yaw = rotationMatrixToEulerAngles(Rot_Mat_Marker)

        tvec_str = "id:%4d x=%4.0f y=%4.0f z=%4.0f pitch=%4.0f roll=%4.0f yaw=%4.0f" % (
            ids[0], T_Vec_3D[0], T_Vec_3D[1], T_Vec_3D[2], math.degrees(pitch), math.degrees(roll),
            math.degrees(yaw))
        cv2.putText(gray_frame, tvec_str, (20, 460 - 20 * 0), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    #Bereken cam 2
    if Max_Cameras == 2:
        if ids2 is not None:
            # Teken een vierkant om de gedetecteerde marker
            aruco.drawDetectedMarkers(gray_frame2, corners2, ids2)
            # Bepaal de positie van 
            rvec_list_all, tvec_list_all, _objPoints = aruco.estimatePoseSingleMarkers(corners2, marker_size, camera_matrix,
                                                                                       camera_distortion)
            rvec2 = rvec_list_all[0][0]
            tvec2 = tvec_list_all[0][0]
            
            cv2.drawFrameAxes(gray_frame2, camera_matrix, camera_distortion, rvec2, tvec2, 100)
            
            # Bepaal rotatiematrix op basis van rotatievector
            rotation_matrix2, jacobian = cv2.Rodrigues(rvec2)
            
            # Maak 3D vector voor translaties
            Vec_3D2 = tvec2 - Vec_3D02
            
            # Roteer de gemeten verplaatsing van V_3D-V_3D0 van camera stelsel naar markerstelsel
            T_Vec_3D2 = np.dot(Vec_3D2, rot_mat02)
            
            # Bereken de rotatie van marker tov markerstelse
            Rot_Mat_Marker2 = np.matmul(rot_mat02.T,rotation_matrix2)
                
            roll2, pitch2, yaw2 = rotationMatrixToEulerAngles(Rot_Mat_Marker2)
            
            tvec_str2 = "id:%4d x=%4.0f y=%4.0f z=%4.0f pitch=%4.0f roll=%4.0f yaw=%4.0f" % (
                ids2[0], T_Vec_3D2[0], T_Vec_3D2[1], T_Vec_3D2[2], math.degrees(pitch2), math.degrees(roll2),
                math.degrees(yaw2))
            cv2.putText(gray_frame2, tvec_str2, (20, 460 - 20 * 0), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    if ShowImage == True:
        cv2.imshow('frame', gray_frame)
        if Max_Cameras==2:
            cv2.imshow("frame2", gray_frame2)



    # Detecteer 0-punt opnieuw met de 'n'-toets
    if keyboard.is_pressed('n'):
        x0, y0, z0, rot_mat0, HaveOrigin = DetectOriginByMarker(cap, aruco_dict, camera_matrix, camera_distortion)
        Vec_3D0 = np.array([x0, y0, z0])
        
        if Max_Cameras == 2:
            x02, y02, z02, rot_mat02, HaveOrigin = DetectOriginByMarker(cap_2, aruco_dict, camera_matrix, camera_distortion)
            Vec_3D02 = np.array([x02, y02, z02])

    # Start met de 's'-toets
    if keyboard.is_pressed('s') and HaveOrigin: 
        Mainloop = True
        break

    # Stopconditie
    cv2.waitKey(1)
    # Stop met de 'q'-toets
    if keyboard.is_pressed('q'):
        key = ord('q')
        Mainloop = False
        break

#___________________________________MAIN_LOOP__________________________________
# Vanaf hier worden de gemeten gegevens opgeslagen in arrays en csv
if Mainloop:
    if Plotting != 0:
        Translatieplot(T, x, y, z)

    start = time.time()
    c = 0
    print("Measurement started")
    while True:
        # Pak een frame en vertk tot een grijstinten frame
        grabResult = cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        gray_frame = grabResult.Array
        if Max_Cameras == 2:
            grabResult2 = cap_2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            gray_frame2 = grabResult2.Array 


        # Record de tijd
        Measured_Time = time.time() - start

        # Vind alle hoeken en markers in frame
        corners, ids, rejected = aruco.detectMarkers(gray_frame, aruco_dict, camera_matrix, camera_distortion)
        if Max_Cameras == 2 and ids is not None and ids2 is not None:
            corners2, ids2, rejected = aruco.detectMarkers(gray_frame2, aruco_dict, camera_matrix, camera_distortion)
        
        if ids is not None:
            # Teken een vierkant om de marker
            aruco.drawDetectedMarkers(gray_frame, corners, ids)
            # Bepaal de positie van alle markers
            rvec_list_all, tvec_list_all, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size,
                                                                                       camera_matrix, camera_distortion)
            for idx, i in enumerate(ids):
                rvec = rvec_list_all[idx][0]
                tvec = tvec_list_all[idx][0]

                cv2.drawFrameAxes(gray_frame, camera_matrix, camera_distortion, rvec, tvec, 100)

                # Bepaal rotatiematrix op basis van rotatievector
                rotation_matrix, jacobian = cv2.Rodrigues(rvec)

                # Maak 3D vector voor translaties
                Vec_3D = tvec - Vec_3D0

                # Roteer de gemeten verplaatsing van V_3D-V_3D0 van camera stelsel naar markerstelsel
                T_Vec_3D = np.dot(Vec_3D, rot_mat0)

                # Bereken de rotatie van marker tov markerstelsel
                Rot_Mat_Marker = np.matmul(rot_mat0.T,rotation_matrix)

                roll, pitch, yaw = rotationMatrixToEulerAngles(Rot_Mat_Marker)
                
                tvec_str = "id:%4d x=%4.0f y=%4.0f z=%4.0f pitch=%4.0f roll=%4.0f yaw=%4.0f" % (
                    ids[idx], T_Vec_3D[0], T_Vec_3D[1], T_Vec_3D[2], math.degrees(pitch), math.degrees(roll),
                    math.degrees(yaw))
                cv2.putText(gray_frame, tvec_str, (20, 460 - 20 * 0), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

        #Bereken cam 2
        if Max_Cameras ==2:
            if ids2 is not None:
                # Teken een vierkant om de gedetecteerde marker
                aruco.drawDetectedMarkers(gray_frame2, corners2, ids2)
                # Bepaal de positie van 
                rvec_list_all, tvec_list_all, _objPoints = aruco.estimatePoseSingleMarkers(corners2, marker_size, camera_matrix,
                                                                                           camera_distortion)
                rvec2 = rvec_list_all[0][0]
                tvec2 = tvec_list_all[0][0]
                
                cv2.drawFrameAxes(gray_frame2, camera_matrix, camera_distortion, rvec2, tvec2, 100)
                
                # Bepaal rotatiematrix op basis van rotatievector
                rotation_matrix2, jacobian = cv2.Rodrigues(rvec2)
                
                # Maak 3D vector voor translaties
                Vec_3D2 = tvec2 - Vec_3D02
                
                # Roteer de gemeten verplaatsing van V_3D-V_3D0 van camera stelsel naar markerstelsel
                T_Vec_3D2 = np.dot(Vec_3D2, rot_mat02)
                
                # Bereken de rotatie van marker tov markerstelse
                Rot_Mat_Marker2 = np.matmul(rot_mat02.T,rotation_matrix2)
                    
                roll2, pitch2, yaw2 = rotationMatrixToEulerAngles(Rot_Mat_Marker2)
                
                tvec_str2 = "id:%4d x=%4.0f y=%4.0f z=%4.0f pitch=%4.0f roll=%4.0f yaw=%4.0f" % (
                    ids2[0], T_Vec_3D2[0], T_Vec_3D2[1], T_Vec_3D2[2], math.degrees(pitch2), math.degrees(roll2),
                    math.degrees(yaw2))
                cv2.putText(gray_frame2, tvec_str2, (20, 460 - 20 * 0), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        
        if Max_Cameras == 2:
            if ids is not None and not ids2 is not None:
                # Data record
                x = np.append(x, [T_Vec_3D[0]])
                y = np.append(y, [T_Vec_3D[1]])
                z = np.append(z, [T_Vec_3D[2]])
                
                yaw_data = np.append(yaw_data, [yaw])
                pitch_data = np.append(pitch_data, [pitch])
                roll_data = np.append(roll_data, [roll])
                
                T = np.append(T, [Measured_Time])
            
            if ids2 is not None and not ids is not None:
                # Data record
                x = np.append(x, [T_Vec_3D2[0]])
                y = np.append(y, [T_Vec_3D2[1]])
                z = np.append(z, [T_Vec_3D2[2]])
                
                yaw_data = np.append(yaw_data, [yaw2])
                pitch_data = np.append(pitch_data, [pitch2])
                roll_data = np.append(roll_data, [roll2])
                
                T = np.append(T, [Measured_Time])
                
            if ids is not None and ids2 is not None:
                # Data record
                x = np.append(x, [(T_Vec_3D[0]+T_Vec_3D2[0])/2])
                y = np.append(y, [(T_Vec_3D[1]+T_Vec_3D2[1])/2])
                z = np.append(z, [(T_Vec_3D[2]+T_Vec_3D2[2])/2])
                
                yaw_data = np.append(yaw_data, [(yaw+yaw2)/2])
                pitch_data = np.append(pitch_data, [(pitch+pitch2)/2])
                roll_data = np.append(roll_data, [(roll+roll2)/2])
                
                T = np.append(T, [Measured_Time])
        else:
            if ids is not None:
                # Data record
                x = np.append(x, [T_Vec_3D[0]])
                y = np.append(y, [T_Vec_3D[1]])
                z = np.append(z, [T_Vec_3D[2]])
                
                yaw_data = np.append(yaw_data, [yaw])
                pitch_data = np.append(pitch_data, [pitch])
                roll_data = np.append(roll_data, [roll])
                
                T = np.append(T, [Measured_Time])

        if ShowImage == True:
            cv2.imshow('frame', gray_frame)
            if Max_Cameras==2:
                cv2.imshow("frame2", gray_frame2)
        
        if len(x) >= data_limit:
            SaveDataToCSV("CSV data\Rawdata",T, x, y, z, roll_data, pitch_data, yaw_data, c)
            c += 1
            T, x, y, z, roll_data, pitch_data, yaw_data = ResetDataArrays(T, x, y, z, roll_data, pitch_data, yaw_data)

        # Data plotting
        if Plotting == 1:
            plt.cla()
            Translatieplot(T, x, y, z)
        if Plotting == 2:
            plt.cla()
            Rotatieplot(T, yaw_data, roll_data, pitch_data)

        # Stopconditie 1
        if time.time() - start >= EndT:
            SaveDataToCSV("CSV data\Rawdata",T, x, y, z, roll_data, pitch_data, yaw_data, c)
            break
        # Stopconditie 2
        if keyboard.is_pressed('q'):
            EndT = time.time() - start
            SaveDataToCSV("CSV data\Rawdata",T, x, y, z, roll_data, pitch_data, yaw_data, c)
            break
        
        cv2.waitKey(1) & 0xFF

#____________________________SYSTEM_WRAP_UP____________________________________    
# Stop cameras
cap.StopGrabbing()
cap.Close()
if Max_Cameras ==2:
    cap_2.StopGrabbing()
    cap_2.Close()
  

cv2.destroyAllWindows()

if Tijdsonderzoek and Mainloop:
    Interval(T, EndT)

Translatieplot(T, x, y, z)
plt.show()
Rotatieplot(T, yaw_data, roll_data, pitch_data)
plt.show()

# _______________________________DATA_PROCCESSING______________________________
if Mainloop:
    for i in range(0, c + 1):
        T, x, y, z, roll_data, pitch_data, yaw_data = ReadCSVToArrays(f"CSV data\Rawdata{i}.csv")
        
        speed_x, T2 = CalculateDerivative(x, T)
        acc_x, T3 = CalculateDerivative(speed_x, T2)
        
        speed_y, T2 = CalculateDerivative(y, T)
        acc_y, T3 = CalculateDerivative(speed_y, T2)
        
        speed_z, T2 = CalculateDerivative(z, T)
        acc_z, T3 = CalculateDerivative(speed_z, T2)
        
        speed_yaw, T2 = CalculateDerivative(yaw_data, T)
        acc_yaw, T3 = CalculateDerivative(speed_yaw, T2)
        
        speed_pitch, T2 = CalculateDerivative(pitch_data, T)
        acc_pitch, T3 = CalculateDerivative(speed_pitch, T2)
        
        speed_roll, T2 = CalculateDerivative(roll_data, T)
        acc_roll, T3 = CalculateDerivative(speed_roll, T2)
        
        SaveProcesedDataToCSV("CSV data\Processed_data",T, x, y, z, roll_data, pitch_data, yaw_data, speed_x, speed_y, speed_z, speed_roll,
                              speed_pitch, speed_yaw,
                              acc_x, acc_y, acc_z, acc_roll, acc_pitch, acc_yaw, i)

        if Plot_speed:
            plt.plot(T2, speed_x, label="x")
            plt.plot(T2, speed_y, label="y")
            plt.plot(T2, speed_z, label="z")
            plt.title("Speed translaties")
            plt.xlabel("t (s)")
            plt.ylabel("speed (-/s)")
            plt.grid(True)
            plt.legend()
            plt.show()
            
        if Plot_acc:
            plt.plot(T3, acc_x, label="x")
            plt.plot(T3, acc_y, label="y")
            plt.plot(T3, acc_z, label="z")
            plt.title("Acceleratie translaties")
            plt.xlabel("t (s)")
            plt.ylabel("acc (-/s^2)")
            plt.grid(True)
            plt.legend()
            plt.show()

        if Plot_speed:
            plt.plot(T2, speed_yaw, label="yaw")
            plt.plot(T2, speed_pitch, label="pitch")
            plt.plot(T2, speed_roll, label="roll")
            plt.title("Speed rotaties")
            plt.xlabel("t (s)")
            plt.ylabel("speed (rad/s)")
            plt.grid(True)
            plt.legend()
            plt.show()
            
        if Plot_acc:
            plt.plot(T3, acc_yaw, label="yaw")
            plt.plot(T3, acc_pitch, label="pitch")
            plt.plot(T3, acc_roll, label="roll")
            plt.title("Acceleratie rotaties")
            plt.xlabel("t (s)")
            plt.ylabel("acc (rad/s^2)")
            plt.grid(True)
            plt.legend()
            plt.show()

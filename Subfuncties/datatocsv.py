"""
In dit script staan de functies voor het opslaan en lezen van data in csv formaat.
Dit script wordt aangeroepen door de main code.
"""

import numpy as np
import pandas as pd

def SaveDataToCSV(path,time, t_x, t_y, t_z, r_x, r_y, r_z, c):
    # Input: arrays of time,  x, y, z, roll, pitch, yaw
    data = np.array([time, t_x, t_y, t_z, r_x, r_y, r_z])
    df = pd.DataFrame(data.T, columns = ['Time (s)','Translation X','Translation Y','Translation z','Rotation Roll (Rad)','Rotation Pitch (Rad)','Rotation Yaw (Rad)'])
    df.to_csv(f"{path}{c}.csv", index=False)
    print("Data saved to CSV")
    
def ResetDataArrays(time, t_x, t_y, t_z, r_x, r_y, r_z):
    time = np.array([])
    t_x = np.array([])
    t_y = np.array([])
    t_z = np.array([])
    r_x = np.array([])
    r_y = np.array([])
    r_z = np.array([])
    print("Data Arrays reset")
    return time, t_x, t_y, t_z, r_x, r_y, r_z

def ReadCSVToArrays(path):
    df = pd.read_csv(path, index_col=None)
    data = df.to_numpy()
    data = data.T
    time = data[0]
    t_x = data[1]
    t_y = data[2]
    t_z = data[3]
    r_x = data[4]
    r_y = data[5]
    r_z = data[6]
    return time, t_x, t_y, t_z, r_x, r_y, r_z


#LENGT VAN SPEED EN ACC VERSCHILD
def SaveProcesedDataToCSV(path,time, t_x, t_y, t_z, r_x, r_y, r_z,st_x, st_y, st_z, sr_x, sr_y, sr_z, at_x, at_y, at_z, ar_x, ar_y, ar_z, c):
    speed_0 = np.array([0])
    acc_0 = np.array([0,0])
    
    st_x = np.append(speed_0, st_x)
    st_y = np.append(speed_0, st_y)
    st_z = np.append(speed_0, st_z)
    sr_x = np.append(speed_0, sr_x)
    sr_y = np.append(speed_0, sr_y)
    sr_z = np.append(speed_0, sr_z)
    
    at_x = np.append(acc_0, at_x)
    at_y = np.append(acc_0, at_y)
    at_z = np.append(acc_0, at_z)
    ar_x = np.append(acc_0, ar_x)
    ar_y = np.append(acc_0, ar_y)
    ar_z = np.append(acc_0, ar_z)
    
    data = np.array([time, t_x, t_y, t_z, r_x, r_y, r_z,st_x, st_y, st_z, sr_x, sr_y, sr_z, at_x, at_y, at_z, ar_x, ar_y, ar_z])
    df = pd.DataFrame(data.T, columns = ['Time (s)','Translation X','Translation Y','Translation z','Rotation Roll (Rad)','Rotation Pitch (Rad)',
                                         'Rotation Yaw (Rad)','Speed x','Speed y','Speed z','Speed roll (Rad/s)','Speed pitch (Rad/s)','Speed yaw (Rad/s)',
                                         "Acceleration x","Acceleration y","Acceleration z","Acceleration roll (Rad/s/s)","Acceleration pitch (Rad/s/s)",
                                         "Acceleration yaw (Rad/s/s)"])
    df.to_csv(f"{path}{c}.csv", index=False)
    print("Procesed Data saved to CSV")
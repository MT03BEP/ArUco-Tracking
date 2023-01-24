# Hier wordt het verwerken van data makkelijker gemaakt. 
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from Subfuncties.datatocsv import SaveDataToCSV


def CombineCSVFiles(path, datatype):
    # Risicovol als het geheugen van de laptop volraakt bij teveel data
    c = 0
    names = glob.glob(f"CSV data\{datatype}*.csv")
    for name in names:
        if c == 0:
            df0 = pd.read_csv(f"{name}", index_col=None)
        else:
            df = pd.read_csv(f"{name}", index_col=None)
            df0 = df0.append(df)
        c+=1
    df0.to_csv(f"CSV data\Completedata_{datatype}.csv", index=False)

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


def CalculateDataParametersFromRaw(fname):
    T,x,y,z,rx,ry,rz = ReadCSVToArrays(fname)
    i = 0
    print(fname)
    names = ["T","x","y","z","rx","ry","rz"]
    data = [T,x,y,z,rx,ry,rz]
    for Type in data:
        mean = np.mean(Type)
        standard_dev = np.std(Type)        
        fig = plt.figure(figsize=(15,10))
        plt.hist(Type)
        plt.title(names[i])
        plt.show(fig)
        print(f"The mean of {names[i]} = {mean}")
        print(f"The standard deveation of {names[i]} = {standard_dev}")
        i+=1
    print("FPS: ",len(T)/T[-1])
    print("---------------------------------------------------------------------")

def SmoothData(data_array,filter_array):
    data = data_array.copy()
    if len(data) < len(filter_array):
        raise ValueError("filter_array is larger than data_array. Can not work")
        
    if len(filter_array)%2 == 0:
        raise ValueError("filter_array is even. Not yet implemented")
    else:
        missing_points = len(filter_array) - 1
        data_1 = data[:int((missing_points/2))]
        data_2 = np.convolve(data,filter_array/(sum(filter_array)),mode='valid')
        data_3 = data[-int(missing_points/2):]
        data_intermediate = np.append(data_1,data_2)
        return np.append(data_intermediate,data_3)

def SmoothDataOnRaw(path,filter_array):
    time, t_x, t_y, t_z, r_x, r_y, r_z = ReadCSVToArrays(path)
    st_x = SmoothData(t_x,filter_array)
    st_y = SmoothData(t_y,filter_array)
    st_z = SmoothData(t_z,filter_array)
    sr_x = SmoothData(r_x,filter_array)
    sr_y = SmoothData(r_y,filter_array)
    sr_z = SmoothData(r_z,filter_array)
    return time, st_x, st_y, st_z, sr_x, sr_y, sr_z


if __name__ == "__main__":
    #CombineCSVFiles("CSV data", "Rawdata")
    paths = [r"C:\Users\31620\BEP-RaspberryPi\Product (Laptop)\Relation size 2 test 1.csv"]
    for i in paths:
        CalculateDataParametersFromRaw(i)
    #CalculateDataParametersFromRaw("CSV data\Completedata_Rawdata.csv")
    #time, st_x, st_y, st_z, sr_x, sr_y, sr_z = SmoothDataOnRaw("CSV data\Rawdata0.csv",np.array([1,1,1,1,1,1,1,1,1]))
    #plt.plot(time,st_x,label="x")
    #plt.plot(time,st_y,label="y")
    #plt.plot(time,st_z,label="z")
    #plt.legend()
    #plt.grid()
    #SaveDataToCSV(time, st_x, st_y, st_z, sr_x, sr_y, sr_z,1)
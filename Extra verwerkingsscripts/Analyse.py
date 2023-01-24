"""
Dit is de file om te runnen na de Main_Code. Deze zal de csv files combineren,
dataset opschonen en plotten
"""

#___________________________________IMPORT_____________________________________
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from Subfuncties.bereken_bewegingen import CalculateDerivative
from Subfuncties.tijdsonderzoek import Interval
from Subfuncties.plotting import Translatieplot, Rotatieplot
from Subfuncties.datatocsv import SaveDataToCSV, ResetDataArrays, ReadCSVToArrays, SaveProcesedDataToCSV
from data_postprocess import *
from Subfuncties.plotting import *

#______________________________Combine datasets________________________________
CombineCSVFiles("../CSV data", "Rawdata")


# Load datasets (test)
T, x, y, z, rx, ry, rz = ReadCSVToArrays("CSV data/Completedata_Rawdata.csv")

#____________________Plotten met outliers, translaties_________________________
# create figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)
fig.set_figheight(6)
fig.set_figwidth(12)

# plot data
ax1.plot(T, x, color = 'blue')
ax2.plot(T, y, color = 'green')
ax3.plot(T, z, color = 'red')
ax1.grid()
ax2.grid()
ax3.grid()

# set axis limits
ax1.set_xlim(0,max(T))
ax1.set_ylim(min(x),max(x))
ax2.set_xlim(0,max(T))
ax2.set_ylim(min(y),max(y))
ax3.set_xlim(0,max(T))
ax3.set_ylim(min(z),max(z))

# set titels and labels
ax1.set_title('X-richting')
ax2.set_title('Y-richting')
ax3.set_title('Z-richting')
fig.supxlabel('Time (s)')
fig.supylabel('Translatie (mm)')

plt.show()

#______________________Plotten met outliers, rotaties__________________________
# create figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)
fig.set_figheight(6)
fig.set_figwidth(12)

# plot data
ax1.plot(T, ry, color='cyan')
ax2.plot(T, rx, color='yellow')
ax3.plot(T, rz, color='magenta')
ax1.grid()
ax2.grid()
ax3.grid()

# set axis limits
ax1.set_xlim(0,max(T))
ax1.set_ylim(min(ry),max(ry))
ax2.set_xlim(0,max(T))
ax2.set_ylim(min(rx),max(rx))
ax3.set_xlim(0,max(T))
ax3.set_ylim(min(rz),max(rz))

# set titels and labels
ax1.set_title('Roll hoek')
ax2.set_title('Pitch hoek')
ax3.set_title('Yaw hoek')
fig.supxlabel('Time (s)')
fig.supylabel('Rotaties (deg)')

plt.show()

# Remove outliers
outlier = np.array([0])
T1 = np.array([])
x1 = np.array([])
y1 = np.array([])
z1 = np.array([])
rx1 = np.array([])
ry1 = np.array([])
rz1 = np.array([])
Rad_max = 0.15
Trans_max = 20
for j in range(0, len(T)):
    dT = T[j] - T[j - 1]
    if dT > 0.01:
        T1 = np.append(T1, T[j])
        x1 = np.append(x1, x[j])
        y1 = np.append(y1, y[j])
        z1 = np.append(z1, z[j])
        rx1 = np.append(rx1, rx[j])
        ry1 = np.append(ry1, ry[j])
        rz1 = np.append(rz1, rz[j])

x2 = x1
y2 = y1
z2 = z1
rx2 = rx1
ry2 = ry1
rz2 = rz1
for i in range(0, len(T1)):
    dT = T1[i]-T1[i-1]
    if abs(ry2[i]-ry2[i-1]) > Rad_max or abs(z2[i]-z2[i-1]) > Trans_max or abs(rx2[i]-rx2[i-1]) > Rad_max:
        if max(outlier) == (i-1):
            x2[i] = x2[i - 1]
            y2[i] = y2[i - 1]
            z2[i] = z2[i - 1]
            rx2[i] = rx2[i - 1]
            ry2[i] = ry2[i - 1]
            rz2[i] = rz2[i - 1]
        else:
            x2[i] = x2[i-1]*2-x2[i-2]
            y2[i] = y2[i-1]*2-y2[i-2]
            z2[i] = z2[i-1]*2-z2[i-2]
            rx2[i] = rx2[i-1]*2-rx2[i-2]
            ry2[i] = ry2[i-1]*2-ry2[i-2]
            rz2[i] = rz2[i-1]*2-rz2[i-2]
        y1[i] = y1[i - 1]
        z1[i] = z1[i - 1]
        rx1[i] = rx1[i - 1]
        ry1[i] = ry1[i - 1]
        rz1[i] = rz1[i - 1]
        outlier = np.append(outlier, i)
        
print(outlier)
EndT = max(T1)
Interval(T1,EndT)

# Calculate speed and accelerations
speed_x, T2 = CalculateDerivative(x2, T1)
acc_x, T3 = CalculateDerivative(speed_x, T2)

speed_y, T2 = CalculateDerivative(y2, T1)
acc_y, T3 = CalculateDerivative(speed_y, T2)

speed_z, T2 = CalculateDerivative(z2, T1)
acc_z, T3 = CalculateDerivative(speed_z, T2)

speed_roll, T2 = CalculateDerivative(rx2, T1)
acc_roll, T3 = CalculateDerivative(speed_roll, T2)

speed_pitch, T2 = CalculateDerivative(ry2, T1)
acc_pitch, T3 = CalculateDerivative(speed_pitch, T2)

speed_yaw, T2 = CalculateDerivative(rz2, T1)
acc_yaw, T3 = CalculateDerivative(speed_yaw, T2)

#_____________________Plotten zonder outliers, translaties_____________________
# create figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)
fig.set_figheight(6)
fig.set_figwidth(12)

# plot data
ax1.plot(T1, x2, color = 'blue')
ax2.plot(T1, y2, color = 'green')
ax3.plot(T1, z2, color = 'red')
ax1.grid()
ax2.grid()
ax3.grid()

# set axis limits
ax1.set_xlim(0,max(T1))
ax1.set_ylim(min(x2),max(x2))
ax2.set_xlim(0,max(T1))
ax2.set_ylim(min(y2),max(y2))
ax3.set_xlim(0,max(T1))
ax3.set_ylim(min(z2),max(z2))

# set titels and labels
ax1.set_title('X-richting')
ax2.set_title('Y-richting')
ax3.set_title('Z-richting')
fig.supxlabel('Time (s)')
fig.supylabel('Translatie (mm)')

plt.show()


#____________________Plotten zonder outliers, rotaties_________________________
# create figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)
fig.set_figheight(6)
fig.set_figwidth(12)

# plot data
ax1.plot(T1, ry2, color='cyan')
ax2.plot(T1, rx2, color='yellow')
ax3.plot(T1, rz2, color='magenta')
ax1.grid()
ax2.grid()
ax3.grid()

# set axis limits
ax1.set_xlim(0,max(T1))
ax1.set_ylim(min(ry2),max(ry2))
ax2.set_xlim(0,max(T1))
ax2.set_ylim(min(rx2),max(rx2))
ax3.set_xlim(0,max(T1))
ax3.set_ylim(min(rz2),max(rz2))

# set titels and labels
ax1.set_title('Roll hoek')
ax2.set_title('Pitch hoek')
ax3.set_title('Yaw hoek')
fig.supxlabel('Time (s)')
fig.supylabel('Rotaties (deg)')

plt.show()

#________________________Moving average filter_________________________________
for i in range(0, len(T1)-2):
    x2[i] = (x2[i]+x2[i-1]+x2[i+1]+x2[i-2]+x2[i+2])/5
    y2[i] = (y2[i-1]+y2[i]+y2[i+1]+y2[i-2]+y2[i+2])/5
    z2[i] = (z2[i-1]+z2[i]+z2[i+1]+z2[i-2]+z2[i+2])/5
    rx2[i] = (rx2[i-1]+rx2[i]+rx2[i+1]+rx2[i-2]+rx2[i+2])/5
    ry2[i] = (ry2[i-1]+ry2[i]+ry2[i+1]+ry2[i-2]+ry2[i+2])/5
    rz2[i] = (rz2[i-1]+rz2[i]+rz2[i+1]+rz2[i-2]+rz2[i+2])/5

# Plotten
#________________________Moving average filter plot____________________________
# TRANSLATIES
# create figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)
fig.set_figheight(6)
fig.set_figwidth(12)

# plot data
ax1.plot(T1, x2, color = 'blue')
ax2.plot(T1, y2, color = 'green')
ax3.plot(T1, z2, color = 'red')
ax1.grid()
ax2.grid()
ax3.grid()

# set axis limits
ax1.set_xlim(0,max(T1))
ax1.set_ylim(min(x2),max(x2))
ax2.set_xlim(0,max(T1))
ax2.set_ylim(min(y2),max(y2))
ax3.set_xlim(0,max(T1))
ax3.set_ylim(min(z2),max(z2))

# set titels and labels
ax1.set_title('X-richting')
ax2.set_title('Y-richting')
ax3.set_title('Z-richting')
fig.supxlabel('Time (s)')
fig.supylabel('Translatie (mm)')

plt.show()

# ROTATIES
# create figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)
fig.set_figheight(6)
fig.set_figwidth(12)

# plot data
ax1.plot(T1, ry2, color='cyan')
ax2.plot(T1, rx2, color='yellow')
ax3.plot(T1, rz2, color='magenta')
ax1.grid()
ax2.grid()
ax3.grid()

# set axis limits
ax1.set_xlim(0,max(T1))
ax1.set_ylim(min(ry2),max(ry2))
ax2.set_xlim(0,max(T1))
ax2.set_ylim(min(rx2),max(rx2))
ax3.set_xlim(0,max(T1))
ax3.set_ylim(min(rz2),max(rz2))

# set titels and labels
ax1.set_title('Roll hoek')
ax2.set_title('Pitch hoek')
ax3.set_title('Yaw hoek')
fig.supxlabel('Time (s)')
fig.supylabel('Rotaties (deg)')

plt.show()

# Let's try to do some post processing with a representive dataset
# import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load datasets
df_ruis = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Datasetruis.csv')
df_rustig = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Datasetrustigegolven.csv')
df_storm = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Datasetstorm.csv')
df_static_test1 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Static 2 test 1 translaties.csv')
df_static_test2 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Static 2 test 2 translaties.csv')
df_static_test3 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Static 2 test 3 translaties.csv')
df_static_test4 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Static 2 test 4 translaties.csv')
df_static_test5 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Static 2 test 5 translaties.csv')
df_static_test6 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Static 2 test 6 translaties.csv')
df_gevoeligheid_test11 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\cameragevoeligheid\Camera gevoeligheid 1 test 1.csv')
df_gevoeligheid_test12 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\cameragevoeligheid\Camera gevoeligheid 1 test 2.csv')
df_gevoeligheid_test21 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\cameragevoeligheid\Camera gevoeligheid 2 test 1.csv')
df_gevoeligheid_test22 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\cameragevoeligheid\Camera gevoeligheid 2 test 2.csv')
df_gevoeligheid_test31 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\cameragevoeligheid\Camera gevoeligheid 3 test 1.csv')
df_gevoeligheid_test32 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\cameragevoeligheid\Camera gevoeligheid 3 test 2.csv')
df_rot_test1 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\static rot 1 test 1.csv')
df_rot_test2 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\static rot 1 test 2.csv')
df_rot_test3 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\static rot 2 test 1.csv')
df_rot_test4 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\static rot 2 test 2.csv')
df_rot_combo1 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Static rot comb 1.csv')
df_rot_combo2 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\static rot comb 2.csv')
df_hexa_x = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Hexapod\X beweging\test 1\X beweging test 1.csv')
df_hexa_y = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Hexapod\Y beweging\test 1\Y beweging test 1.csv')
df_hexa_z = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Hexapod\Z beweging\test 1\Z beweging test 1.csv')
#df_hexa_xrot = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Hexapod\X rotaties\test 1\X beweging test 1.csv')
df_hexa_yrot = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Hexapod\Y rotaties\test 1\Y rotatie test 1.csv')
df_hexa_zrot = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Hexapod\Z rotaties\test 1\Z rotatie test 1.csv')
df_hexa_golf1 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Hexapod\Golf 1\Rawdata0.csv')
df_hexa_golf2 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Hexapod\Golf 2\Rawdata0.csv')
df_hexa_combi = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Hexapod\Combi rotaties\Rawdata0.csv')
df_hexa_test1 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\Hexapod\Y beweging\test 3 schijnbaar\Y beweging test 3.csv')

df_trans_x1 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\translatie\Actuator X 1.csv')
df_trans_x2 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\translatie\Actuator X 2.csv')
df_trans_y1 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\translatie\Actuator Y 1.csv')
df_trans_y2 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\translatie\Actuator Y 2.csv')
df_trans_z1 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\translatie\Actuator Z 1.csv')
df_trans_z2 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\translatie\Actuator Z 2.csv')

# define dataset to analyze
df = df_hexa_zrot


# write to arrays
time = df['Time (s)'].to_numpy()
x = df['Translation X'].to_numpy()
y = df['Translation Y'].to_numpy()
z = df['Translation z'].to_numpy()
pitch_rad = df['Rotation Pitch (Rad)'].to_numpy()
roll_rad = df['Rotation Roll (Rad)'].to_numpy()
yaw_rad = df['Rotation Yaw (Rad)'].to_numpy()
pitch = np.rad2deg(pitch_rad)
roll = np.rad2deg(roll_rad)
yaw = np.rad2deg(yaw_rad)

# example of movement beunen
# time = np.arange(0,61,1,dtype=int)
# xy_move = int(100)* np.ones(10, dtype=int)
# z_move = int(40)* np.ones(10,dtype=int)
# x1 = np.zeros(6, dtype=int)
# x2 = np.zeros(45, dtype=int)
# y1 = np.zeros(21)
# y2 = np.zeros(30)
# z1 = np.zeros(36)
# z2 = np.zeros(15)
# xtemp = np.append(x1,xy_move)
# x = np.append(xtemp,x2)
# ytemp = np.append(y1,xy_move)
# y = np.append(ytemp,y2)
# ztemp = np.append(z1, z_move)
# z =  np.append(ztemp, z2)
#
# pitch = np.zeros(len(time))
# roll = np.zeros(len(time))
# yaw = np.zeros(len(time))

# determine acceleration


# filter outliers



# print some usefull data
print('Translatie overzicht')
print('max. x-translatie:', max(x), 'mm')
print('min. x-translatie:', min(x), 'mm')
print('max. y-translatie:', max(y), 'mm')
print('min. y-translatie:', min(y), 'mm')
print('max. z-translatie:', max(z), 'mm')
print('min. z-translatie:', min(z), 'mm')


# figure settings - axis limits
time_limit = [0,15]
translatie_limit = [-100,100]
rotatie_limit = [-8,8]


#_________________________________________________________________________
# PLOT TRANSLATIES
#_________________________________________________________________________
# create figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)
fig.set_figheight(12)
fig.set_figwidth(12)

# plot data
ax1.plot(time, x, color='blue')
ax2.plot(time, y, color='green')
ax3.plot(time, z, color='red')
ax1.grid()
ax2.grid()
ax3.grid()

# set axis limits
ax1.set_xlim(time_limit)
ax1.set_ylim(translatie_limit)
ax2.set_xlim(time_limit)
ax2.set_ylim(translatie_limit)
ax3.set_xlim(time_limit)
ax3.set_ylim(translatie_limit)

# set font size of thicks
ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)
ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)

# set titels and labels
ax1.set_title('X-richting', fontsize=24)
ax2.set_title('Y-richting', fontsize=24)
ax3.set_title('Z-richting', fontsize=24)
fig.supxlabel('Time (s)', fontsize=20)
fig.supylabel('Translatie (mm)', fontsize=20)

plt.show()


#_________________________________________________________________________
# PLOT ROTATIES
#_________________________________________________________________________
# create figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)
fig.set_figheight(12)
fig.set_figwidth(12)

# plot data
ax1.plot(time, roll, color='cyan')
ax2.plot(time, pitch, color='yellow')
ax3.plot(time, yaw, color='magenta')
ax1.grid()
ax2.grid()
ax3.grid()

# set axis limits
ax1.set_xlim(time_limit)
ax1.set_ylim(rotatie_limit)
ax2.set_xlim(time_limit)
ax2.set_ylim(rotatie_limit)
ax3.set_xlim(time_limit)
ax3.set_ylim(rotatie_limit)

# set font size of thicks
ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)
ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)

# set titels and labels
ax1.set_title('Roll hoek', fontsize=24)
ax2.set_title('Pitch hoek', fontsize=24)
ax3.set_title('Yaw hoek', fontsize=24)
fig.supxlabel('Time (s)', fontsize=20)
fig.supylabel('Rotaties (deg)', fontsize=20)

plt.show()



#_________________________________________________________________________
# DATA ANALYSE MET PANDAS
#_________________________________________________________________________
df_timeintlow = df[df['Time (s)'] > 12]
df_timeint = df_timeintlow[df_timeintlow['Time (s)'] < 15]
df=df_timeint

print('MEAN')
print(df.mean())
print('Standaar deivatie')
print(df.std())

# df_qauntile  =df.quantile([0.25,0.5,0.75])
# df_mean = df.mean()
# df_min = df.min()
# df_max = df.max()
#
# print(df_qauntile)
# print('MEAN')
# print(df_mean)
# print('MIN')
# print(df_min)
# print('MAX')
# print(df_max)
# df_qauntile.to_clipboard(sep=',', index=False)

# # Translaties boxplot en histogram
# boxplot_translatie = df.boxplot(column=['Translation X', 'Translation Y', 'Translation z'], figsize= (6,4), showmeans=True)
# hist_translatie = df.hist(bins=10, column=['Translation X', 'Translation Y', 'Translation z'], figsize= (6,4))
# #pdf_translatie = df.plot.kde(column=['Translation X', 'Translation Y', 'Translation z'])
# plt.show()
#
# # Rotatie boxplot en histogram
# boxplot_rotatie = df.boxplot(column=['Rotation Pitch (Rad)', 'Rotation Roll (Rad)', 'Rotation Yaw (Rad)'], figsize= (6,4), showmeans=True)
# hist_rotatie = df.hist(bins=10, column=['Rotation Pitch (Rad)', 'Rotation Roll (Rad)', 'Rotation Yaw (Rad)'], figsize= (6,4))
# plt.show()
#
# print(type(boxplot_rotatie))

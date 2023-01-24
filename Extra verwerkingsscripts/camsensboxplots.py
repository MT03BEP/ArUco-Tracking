import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import all frames
df_gevoeligheid_test11 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\cameragevoeligheid\Camera gevoeligheid 1 test 1.csv')
df_gevoeligheid_test12 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\cameragevoeligheid\Camera gevoeligheid 1 test 2.csv')
df_gevoeligheid_test21 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\cameragevoeligheid\Camera gevoeligheid 2 test 1.csv')
df_gevoeligheid_test22 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\cameragevoeligheid\Camera gevoeligheid 2 test 2.csv')
df_gevoeligheid_test31 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\cameragevoeligheid\Camera gevoeligheid 3 test 1.csv')
df_gevoeligheid_test32 = pd.read_csv(r'C:\Users\evasm\Documents\TU_Delft\BEP-algemeen\Data\cameragevoeligheid\Camera gevoeligheid 3 test 2.csv')

# make combined set?
df_webcam = df_gevoeligheid_test12
df_mv = df_gevoeligheid_test21
df_mv2 = df_gevoeligheid_test31

df=df_mv2

# Translaties boxplot en histogram
boxplot_translatie = df.boxplot(column=['Translation X', 'Translation Y', 'Translation z'], figsize= (6,4))
plt.title('Webcam translatie gevoeligheid')
plt.show()
# Rotatie boxplot en histogram
boxplot_rotatie = df.boxplot(column=['Rotation Pitch (Rad)', 'Rotation Roll (Rad)', 'Rotation Yaw (Rad)'], figsize= (6,4))
plt.title('Webcam rotatie gevoeligheid')
plt.show()


df.rename(columns={
    "Translation X": "x",
    "Translation Y": "y",
    "Translation z": "z",
    "Rotation Roll (Rad)": "roll",
    "Rotation Pitch (Rad)": "pitch",
    "Rotation Yaw (Rad)": "yaw"
},
          inplace=True)

boxplot = df.boxplot(column=['x', 'y', 'z'], figsize= (6,4))
plt.title('2x Machine Vision translatie gevoeligheid')
plt.ylim(-20, 20)
plt.ylabel('Translatie (mm)')
plt.show()

boxplot = df.boxplot(column=['roll', 'pitch', 'yaw'], figsize= (6,4))
plt.title('2x Machine Vision rotatie gevoeligheid')
plt.ylabel('Rotatie (rad)')
plt.ylim(-2, 2)
plt.show()
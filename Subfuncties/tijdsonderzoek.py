"""
In dit script staan de functies voor het naberekenen van tijddata. Dit script 
wordt aangeroepen door de main code.
"""

# Testen wat de standaardafwijking is van de tijdsinterval
import numpy as np
import cv2
import cv2.aruco as aruco
import math
import time
import matplotlib.pyplot as plt

def Interval(T, EndT):
    Intvl = T[1:] - T[:-1]
    print("Gemeten tijdintervallen",Intvl)
    binsx = np.linspace(0.00, 0.15, 151)
    plt.hist(Intvl, bins=binsx)
    plt.grid(True)
    plt.show()
    print(f"aantal frames: {len(T)}")
    print(F"aantal fps: {len(T)/(T[-1]-T[0])}")
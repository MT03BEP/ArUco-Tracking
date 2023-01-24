"""
In dit script staan de functies voor het plotten van data. Dit script wordt 
aangeroepen door de main code.
"""

import matplotlib.pyplot as plt

def Translatieplot(T,x,y,z):
    plt.plot(T, x, label="x")
    plt.plot(T, y, label="y")
    plt.plot(T, z, label="z")
    plt.title("Translaties")
    plt.xlabel("t (s)")
    plt.ylabel("x,y,z (-)")
    plt.legend()
    plt.grid(True)
    plt.pause(0.05)

def Rotatieplot(T,yaw_data,roll_data,pitch_data):
    plt.plot(T, yaw_data, label="yaw")
    plt.plot(T, roll_data, label="roll")
    plt.plot(T, pitch_data, label="pitch")
    plt.title("Rotaties")
    plt.xlabel("t (s)")
    plt.ylabel("y,p,r (rad)")
    plt.legend()
    plt.grid(True)
    plt.pause(0.05)


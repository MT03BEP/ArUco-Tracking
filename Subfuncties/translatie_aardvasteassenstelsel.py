"""
In dit script staan de functies voor het verwerken van rotaties. Dit script 
wordt aangeroepen door de main code.
"""

import numpy as np
import math

# Check of matrix een rotatie matrix is
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Rotatiematrix naar Euler hoeken functie
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])

    singular = sy < 1e-6

    if not singular:
        rot_x = math.atan2(R[2, 1], R[2, 2])
        rot_y = math.atan2(-R[2, 0], sy)
        rot_z = math.atan2(R[1, 0], R[0, 0])
    else:
        rot_x = math.atan2(-R[1, 2], R[1, 1])
        rot_y = math.atan2(-R[2, 0], sy)
        rot_z = 0
    return np.array([rot_x, rot_y, rot_z])

# Hoeken naar rotatiematrix ter controle
def EulerAnglesToRotationMatrix(a1,a2,a3):
    R_x = np.array([[1,                    0,              0],
                    [0,         math.cos(a1), -math.sin(a1) ],
                    [0,         math.sin(a1),  math.cos(a1) ]])
 
    R_y = np.array([[ math.cos(a2),    0,      math.sin(a2)],
                    [            0,    1,                 0],
                    [-math.sin(a2),    0,      math.cos(a2)]])
 
    R_z = np.array([[math.cos(a3),    -math.sin(a3),    0],
                    [math.sin(a3),     math.cos(a3),    0],
                    [           0,                0,    1]])
 
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R
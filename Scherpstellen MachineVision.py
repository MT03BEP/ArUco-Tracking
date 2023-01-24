"""
Dit script is bedoelt voor het scherpstellen van Basler MachineVision camera's
op Pypylon. Sluit 1 camera aan op de usb poort en run dit script. In een popup
vesnster kan de camera worden scherpgesteld en de licht inval geregeld worden.
Verlaat het script door q ingedrukt te houden.
"""

#________________________________IMPORTS_______________________________________
import cv2
from pypylon import pylon


frame_factor = 1.5


#_____________________________Camera instellen_________________________________
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera_width = 640
camera_height = 480
camera.Width.SetValue(camera_width)
camera.Height.SetValue(camera_height)
camera_width = int(frame_factor*640)
camera_height = int(frame_factor*480)
camera.CenterX.SetValue(True)
camera.CenterY.SetValue(True)
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
print(f"MV camera width: {camera_width}\nMV camera height: {camera_height}")

#_____________________________Scherpstel deel__________________________________
while True:
    grabResult = camera.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
    gray_frame = grabResult.Array

    cv2.imshow('frame', gray_frame)

    # Stopconditie
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        key = ord('q')
        break

# _____________________________Afsluiten_______________________________________
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import pulsectl

###########################
wCam, hCam = 1280, 720
###########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

# PulseAudio kontrolünü başlatma
pulse = pulsectl.Pulse('volume-script')

# Varsayılan sink (çıkış cihazı) alma
default_sink = pulse.sink_list()[0]

# Mevcut ses seviyesini okuma
volRange = [0.0, 1.0]  # PulseAudio ses seviyesi 0.0 ile 1.0 arasında değişir
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        if len(lmList) > 8:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            vol = np.interp(length, [50, 300], [minVol, maxVol])
            volPer = np.interp(length, [50, 300], [0, 100])
            volBar = np.interp(length, [50, 300], [400, 150])

            pulse.volume_set_all_chans(default_sink, vol)
            print(f'Volume: {volPer}%')


            if length < 45:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv2.putText(img, f'Vol: {int(volPer)}%', (40, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
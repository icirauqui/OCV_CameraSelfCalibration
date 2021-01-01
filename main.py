import cv2
import numpy as np

cap = cv2.VideoCapture('input.mp4')

if (cap.isOpened()==False):
    print("Error opening video stream or file")

orb = cv2.ORB_create()
matcher = cv2.BFMatcher()
frameid = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frameid = frameid + 1
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF==ord('q'):
            break
            
        

        if frameid==1 or frameid%10==0:
            frame_bw = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            kps, dsc = orb.detectAndCompute(frame_bw,None)

            if frameid%10==0:
                matches = matcher.match(dsc1,dsc)
                framematched = cv2.drawMatches(frame1,kps1,frame,kps,matches[:20],None)
                cv2.imshow("Matches",framematched)


            
            kps1=kps
            dsc1=dsc
            frame1=frame

    else:
        break

cap.release()

cv2.destroyAllWindows()


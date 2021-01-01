import cv2
import numpy as np

cap = cv2.VideoCapture('input.mp4')

if (cap.isOpened()==False):
    print("Error opening video stream or file")

orb = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
                matches = sorted(matches, key = lambda x:x.distance)
                framematched = cv2.drawMatches(frame1,kps1,frame,kps,matches[:20],None)
                cv2.imshow("Matches",framematched)

                p1 = []
                p2 = []
                A = []
                
                if len(matches)>=20:
                    for i in range(20):
                        #print(kps1[matches[i].trainIdx].pt," - ",kps[matches[i].queryIdx].pt," - ",matches[i].distance)
                        p1.append(kps1[matches[i].trainIdx].pt)
                        p2.append(kps[matches[i].queryIdx].pt)

                        p11 = kps1[matches[i].trainIdx].pt[0]
                        p12 = kps1[matches[i].trainIdx].pt[1]
                        p21 = kps[matches[i].queryIdx].pt[0]
                        p22 = kps[matches[i].queryIdx].pt[1]

                        # Matriz fundamental
                        #Ai = [p21*p11,p21*p12,p21,p22*p11,p22*p12,p22,p11,p12,1]
                        #A.append(Ai)

                        # Homografía
                        Ai1 = [p11, p12,   1,   0,   0,   0, -p11*p21, -p12*p21, -p21]
                        Ai2 = [  0,   0,   0, p11, p12,   1, -p11*p22, -p12*p22, -p22]
                        A.append(Ai1)
                        A.append(Ai2)

                        

                    U,S,V = np.linalg.svd(A)
                    f = V[:,8]
                    Fn = [[f[0],f[1],f[2]],[f[3],f[4],f[5]],[f[6],f[7],f[8]]]
                    U,S,V = np.linalg.svd(Fn)
                    S[2]=0
                    V = np.transpose(V)
                    Fn2 = U * S * V
                    print(Fn2)



            
            kps1=kps
            dsc1=dsc
            frame1=frame

    else:
        break

cap.release()

cv2.destroyAllWindows()


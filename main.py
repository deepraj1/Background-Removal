import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS,60)
seg = SelfiSegmentation()
fps = cvzone.FPS()

list = os.listdir("img")
imglist = []
for imgPath in list:
    img = cv2.imread(f'img/{imgPath}')
    resizeimg = cv2.resize(img,(640,480))
    imglist.append(resizeimg)

imgindex = 0

while True:
    s,img = cap.read()
    imgout = seg.removeBG(img,imglist[imgindex],threshold=0.8)
    stacked =  cvzone.stackImages([img,imgout],2,0.5)
    _,stacked = fps.update(stacked)
    cv2.imshow("image",stacked)
    print(imgindex)
    key = cv2.waitKey(1)
    if key==ord('b'):
        if imgindex>0:
            imgindex-=1

    if key==ord('n'):
        if imgindex<len(imglist)-1:
            imgindex+=1

    if key == ord('q'):
        break
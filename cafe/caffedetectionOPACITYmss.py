import cv2, numpy as np
thres = 0.3
from mss import mss;sct=mss()
classNames = { 0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus/truck',
               7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
               14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
classColors = { 0: (255,255,255), 1:(255,255,255) , 2: (255,255,255), 3: (255,255,255), 4: (255,255,255), 5: (255,255,255), 6: (255,0,255),
               7: (0,0,255), 8:(255,255,255) , 9: (255,255,255), 10: (255,255,255), 11: (255,255,255), 12: (255,255,255), 13: (255,255,255),
               14: (255,255,255), 15: (0,255,255), 16: (255,255,255), 17: (255,255,255), 18: (255,255,255), 19: (255,255,255),
               20: (255,255,255)}

net = cv2.dnn_DetectionModel("MobileNetSSD_deploy.prototxt","MobileNetSSD_deploy.caffemodel")
net.setInputSize(1220,660);net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5));
net.setInputSwapRB(True)
while True:
    img = np.array(sct.grab(
        {'top': 174, 'left': 126, 'width': 1220, 'height': 660}))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  
    blk = np.zeros(img.shape, np.uint8)
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            if confidence >0.5:
                if box[3]<300:cv2.rectangle(blk,box,classColors[classId],thickness=2)
                if classId!=15:
                    cv2.putText(blk,classNames[classId]+" "+str(int(confidence*100)),(box[0],box[1]+15), cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
                #cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    img = cv2.addWeighted(img, 1, blk, 0.5, 1)
    cv2.imshow("Output2",img)
    cv2.waitKey(1)

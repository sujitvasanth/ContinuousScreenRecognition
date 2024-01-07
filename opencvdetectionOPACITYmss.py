import cv2, numpy as np; from mss import mss;sct=mss()
thres = 0.45 # Threshold to detect object

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
while True:
    img = np.array(sct.grab(
        {'top': 174, 'left': 126, 'width': 610, 'height': 330}))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)    
    #img = cv2.resize(img,(320,200))

    blk = np.zeros(img.shape, np.uint8)
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(blk,box,color=(100,100,255),thickness=2)
            cv2.putText(img,classNames[classId-1],(box[0],box[1]+15),
                        cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
            #cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30)
    img = cv2.addWeighted(img, 1, blk, 0.5, 1)
    cv2.imshow("Output",img)
    cv2.waitKey(1)


import streamlit as st
import urllib3
import numpy as np
from PIL import Image
import cv2
import requests
import socket

#================================
#   Message Headers
#=================================
COMMAND_START=bytes("<command>",'utf-8')
COMMAND_END=bytes("</command>","utf-8")
IMAGE_START=bytes("<image>","utf-8")
IMAGE_END=bytes("</image>","utf-8")

#================================
#   Web App Init Elements
#=================================
st.title("ONVIF CCTV Connect")
st.write("(C) Faizansoft International 2000-2021")
st.write("\r\n")
st.write("Note: This demo will only work with ONVIF compatible IP cameras that have the live-jpeg API.")
st.write("The reason for live jpeg being chosen over rtsp/rtmp is due to reliability on low resourced cameras.")
#=================================
#   Set up Yolo V4 
#=================================
class YoloV4Model:
    def __init__(self,yolocfg,yoloweights,coconames):
        self.CONFIDENCE_THRESHOLD = 0.2
        self.NMS_THRESHOLD=0.4
        
        #Set up neural network and configure Backend and Target
        dnn_net=cv2.dnn.readNetFromDarknet(yolocfg, yoloweights)
        dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        #Set up the DNN Model
        dnn_model=cv2.dnn_DetectionModel(dnn_net)
        dnn_model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
        self._dnn_model=dnn_model
        
        #Setup the coco.names list
        COCO_NAMES_LIST=[]
        with open("coco.names","r") as coco_names:
            COCO_NAMES_LIST=coco_names.readlines()
        self._COCO_NAMES_LIST=COCO_NAMES_LIST
        
    def DetectObjects_retFrameDetList(self,frame):
        _classes,_scores,_boxes=self._dnn_model.detect(frame,self.CONFIDENCE_THRESHOLD,self.NMS_THRESHOLD)
        
        #Text List for detections
        DET_LIST=[]
        
        for (_class,_score,_box) in zip(_classes,_scores,_boxes):
            _class=_class.tolist()
            _score=_score.tolist()
            _box=_box.tolist()
            cv2.rectangle(frame,_box, (0,255,0), 2)
            cv2.putText(frame, self._COCO_NAMES_LIST[_class[0]], (_box[0], _box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,5,(0,255,255), 2)
            DET_LIST.extend("Detected {} @ {},{}.".format(self._COCO_NAMES_LIST[_class[0]],_box[0],_box[1]))
        return (frame,DET_LIST)
_yoloV4=YoloV4Model("yolov4.cfg","yolov4.weights","coco.names")

#================================
#   Getting Camera Variables
#=================================
# Here we try to get the variables for the user's camera.
# This includes ip_addr,uname and password
with st.form(key="ip_cctv_connect"):
    st.write("Please enter the credentials for your ONVIF Capable IP Camera.")
    ip_address=st.text_input("Enter your camera's IP Address:")
    username=st.text_input("Enter your Camera's Username:")
    password=st.text_input("Enter your camera's password:")
    command=st.text_input("Enter the image processing command: ")
    cmd_connect=st.form_submit_button(label="Connect!")
    
#=====================================
#   Disconnect Button
#==========================================
cmd_disconnect=st.button("Disconnect!")
    
#===============================
#   URLLIB 3 HTTP OBject
#===============================
http=urllib3.PoolManager()

#===============================
#   Streamlit Placeholders
#===============================
#Create the Place Holders
img_ph_1=st.image([])
img_ph_2=st.image([])


def grab_frame_cctv():
    #http://admin:admin@192.168.1.10/tmpfs/auto.jpg
    _url="http://{0}:{1}@{2}/tmpfs/auto.jpg".format(username,password,ip_address)
    img=Image.open(requests.get(_url,stream=True).raw)
    cvFrame=np.array(img)
    return cvFrame


if cmd_connect:
    while True:
        frame=grab_frame_cctv()
        img_ph_1.image(frame)
        sock=socket.socket()
        sock.connect(("127.0.0.1",1024))
        sockMSg=bytearray(COMMAND_START)
        sockMSg.extend(bytes(command,'utf-8'))
        sockMSg.extend(COMMAND_END)
        sockMSg.extend(IMAGE_START)
        sockMSg.extend(cv2.imencode(".JPEG",frame)[1])
        sockMSg.extend(IMAGE_END)
        sock.sendall(sockMSg)
        frame=cv2.imdecode(np.frombuffer(sock.recv(999999999),np.uint8),1)
        sock.close()
        img_ph_2.image(frame)
        
        if cmd_disconnect:
            break
    






     
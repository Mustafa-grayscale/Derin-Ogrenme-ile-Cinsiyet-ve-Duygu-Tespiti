import cv2
import numpy as np
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def cnn_Model(face_crop,model,):
    classes = ['man','woman']


    face_crop = cv2.resize(face_crop, (96,96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)
    
    
    conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
    idx = np.argmax(conf)
    label = classes[idx]
    result = "{}: {:.2f}%".format(label, conf[idx] * 100)
    
    return result

img = cv2.imread("images/image/neutral.jpg")
img_width = img.shape[1]
img_height = img.shape[0]

model_CNN = load_model('gender_detection.model')

img_blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB=True, crop=False)

labels = ["Angry","Fear","Happy","Neutral","Sad","Surprise"]

colors = ["0,255,255","0,0,255","255,0,0","255,255,0","0,255,0"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors,(18,1))

model = cv2.dnn.readNetFromDarknet('emotion_yolov4.cfg',
                                  'emotion_yolov4_best.weights')
layers = model.getLayerNames()
output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

detection_layers = model.forward(output_layer)


ids_list = []
boxes_list = []
confidences_list = []



for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]
        
        if confidence > 0.20:
            
            label = labels[predicted_id]
            bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
            
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
            
            start_x = int(box_center_x - (box_width/2))
            start_y = int(box_center_y - (box_height/2))
            
            ids_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
            
            
max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
            
for max_id in max_ids:
    max_class_id = max_id[0]
    box = boxes_list[max_class_id]
    
    start_x = box[0]
    start_y = box[1]
    box_width = box[2]
    box_height = box[3]
    
    
    predicted_id = ids_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidences_list[max_class_id]
    end_x = start_x + box_width
    end_y = start_y + box_height
    
    face_crop = img[start_y : end_y, start_x : end_x]
    result = cnn_Model(face_crop, model_CNN)
            
    box_color = colors[predicted_id]
    box_color = [int(each) for each in box_color]
            
            
    label = "{}: {:.2f}%".format(label,confidence*100)
    print("predicted object {}".format(label))
           
            
           
            
           
    cv2.rectangle(img, (start_x,start_y),(end_x,end_y),box_color,1)
    cv2.putText(img,label + ' '  + result ,(start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            
            
cv2.imshow("Detection Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
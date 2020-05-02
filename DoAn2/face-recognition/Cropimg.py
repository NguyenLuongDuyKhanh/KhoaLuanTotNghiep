import os
import time
import glob
import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K

path='.'
# Get Image names stored in "Images" folder
image_path_names=[]
person_names=set()
for file_name in glob.glob('./Images/*_[0-9]*.jpg'):
    image_path_names.append(file_name)
    person_names.add(image_path_names[-1].split('\\')[-1].split('_')[0])

print(len(image_path_names))
print(person_names)

# Load CNN face detector into dlib
dnnFaceDetector=dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

#os.mkdir(path+'/Images_crop/')


# For each person create a separate folder
for person in person_names:
    os.mkdir(path+'/Images_crop/'+person+'/')
#   detected face and save them in corresponding person folder
for file_name in image_path_names:
    print(file_name)
    img=cv2.imread(file_name)

    # get dimensions of image
    dimensions = img.shape

    print('Image Dimension    : ',dimensions)
    resized = img
    if int(img.shape[1]) > 450 :
        scale_percent = (450*100)/img.shape[1]
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = resized

    dimensions = img.shape

    print('Image Dimension    : ',dimensions)

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects=dnnFaceDetector(gray,1)
    print("break")

    left,top,right,bottom=0,0,0,0
    for (i,rect) in enumerate(rects):
        left=rect.rect.left() #x1
        top=rect.rect.top() #y1
        right=rect.rect.right() #x2
        bottom=rect.rect.bottom() #y2
    width=right-left
    height=bottom-top
    img_crop=img[top:top+height,left:left+width]
    img_path=path+'/Images_crop/'+file_name.split('\\')[-1].split('_')[0]+'/'+file_name.split('\\')[-1]
    cv2.imwrite(img_path,img_crop)

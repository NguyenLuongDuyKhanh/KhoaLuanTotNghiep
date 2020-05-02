import os
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
# Load saved model
classifier_model=tf.keras.models.load_model('face_classifier_model.h5')

# Path to folder which contains images to be tested and predicted
test_images_path=path+'./Test_Images/'
dnnFaceDetector=dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
"""
def plot(img):
  plt.figure(figsize=(8,4))
  plt.imshow(img[:,:,::-1])
  plt.show()
  """
# Label names for class numbers
#person_rep={0:'Narendra Modi',1:'Donald Trump',2:'Angela Merkel',3:'Xi Jinping',4:'Lakshmi Narayana',5:'Vladimir Putin'}
person_rep={0:'Angela Merkel',1:'Xi Jinping',2:'Cha gia kinh yeu',3:'Narendra Modi',4:'Vladimir Putin',5:'Donald Trump'}
os.mkdir(path+'/Predictions')
#Define VGG_FACE_MODEL architecture
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# Load VGG Face model weights
model.load_weights('vgg_face_weights.h5')
model.summary()
vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

for img_name in os.listdir('./Test_Images/'):
    print(img_name)
    if img_name=='crop_img.jpg':
        continue
    # Load Image
    print('Load Image')
    img=cv2.imread(path+'/Test_Images/'+img_name)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect Faces
    print("Detect Faces")
    rects=dnnFaceDetector(gray,1)
    left,top,right,bottom=0,0,0,0
    for (i,rect) in enumerate(rects):     
        # Extract Each Face
        print('Extract Each Face')
        left=rect.rect.left() #x1
        top=rect.rect.top() #y1
        right=rect.rect.right() #x2
        bottom=rect.rect.bottom() #y2
        width=right-left
        height=bottom-top
        img_crop=img[top:top+height,left:left+width]
        cv2.imwrite(path+'/Test_Images/crop_img.jpg',img_crop)
    
        # Get Embeddings
        print('Get Embeddings')
        crop_img=load_img(path+'/Test_Images/crop_img.jpg',target_size=(224,224))
        crop_img=img_to_array(crop_img)
        crop_img=np.expand_dims(crop_img,axis=0)
        crop_img=preprocess_input(crop_img)
        img_encode=vgg_face(crop_img)

        # Make Predictions
        print('Make Predictions')
        embed=K.eval(img_encode)
        person=classifier_model.predict(embed)
        name=person_rep[np.argmax(person)]
        os.remove(path+'/Test_Images/crop_img.jpg')
        cv2.rectangle(img,(left,top),(right,bottom),(0,255,0), 2)
        img=cv2.putText(img,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
        img=cv2.putText(img,str(np.max(person)),(right,bottom+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
        cv2.imshow('hihi',img)
        cv2.waitKey(0)
    # Save images with bounding box,name and accuracy 
    print('Save images with bounding box,name and accuracy')
    cv2.imwrite(path+'/Predictions/'+img_name,img)
    #plot(img)
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

path = '.'
dnnFaceDetector=dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
"""
# Get Image names for testing
test_image_path_names=[]
for file_name in glob.glob(path+'/Images_test/*_[123].jpg'):
    test_image_path_names.append(file_name)

print(len(test_image_path_names))

#os.mkdir(path+'/Test_Images_crop/')
image_path_names=[]
person_names=set()
for file_name in glob.glob('./Images/*_[0-9]*.jpg'):
    image_path_names.append(file_name)
    person_names.add(image_path_names[-1].split('\\')[-1].split('_')[0])

# Create Separate folder for each person in "Test_Images_crop" folder
for person in person_names:
    os.mkdir(path+'/Test_Images_crop/'+person+'/')

# Detect face,crop face and save in corresponding folder
for file_name in test_image_path_names:
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

    left,top,right,bottom=0,0,0,0
    for (i,rect) in enumerate(rects):
        left=rect.rect.left() #x1
        top=rect.rect.top() #y1
        right=rect.rect.right() #x2
        bottom=rect.rect.bottom() #y2
    width=right-left
    height=bottom-top
    img_crop=img[top:top+height,left:left+width]
    img_path=path+'/Test_Images_crop/'+file_name.split('\\')[-1].split('_')[0]+'/'+file_name.split('\\')[-1]
    cv2.imwrite(img_path,img_crop)
    """

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

# Remove Last Softmax layer and get model upto last flatten layer with outputs 2622 units
vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

#Prepare Training Data
x_train=[]
y_train=[]
person_folders=os.listdir(path+'/Images_crop/')
person_rep=dict()
for i,person in enumerate(person_folders):
    person_rep[i]=person
    image_names=os.listdir('Images_crop/'+person+'/')
    for image_name in image_names:
        img=load_img(path+'/Images_crop/'+person+'/'+image_name,target_size=(224,224))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        img=preprocess_input(img)
        img_encode=vgg_face(img)
        x_train.append(np.squeeze(K.eval(img_encode)).tolist())
        y_train.append(i)

person_rep
x_train=np.array(x_train)
y_train=np.array(y_train)

#Prepare Test Data
x_test=[]
y_test=[]
person_folders=os.listdir(path+'/Test_Images_crop/')
for i,person in enumerate(person_folders):
    image_names=os.listdir('Test_Images_crop/'+person+'/')
    for image_name in image_names:
        img=load_img(path+'/Test_Images_crop/'+person+'/'+image_name,target_size=(224,224))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        img=preprocess_input(img)
        img_encode=vgg_face(img)
        x_test.append(np.squeeze(K.eval(img_encode)).tolist())
        y_test.append(i)

print(person_rep)

x_test=np.array(x_test)
y_test=np.array(y_test)

# Save test and train data for later use
np.save('train_data',x_train)
np.save('train_labels',y_train)
np.save('test_data',x_test)
np.save('test_labels',y_test)

# Load saved data
x_train=np.load('train_data.npy')
y_train=np.load('train_labels.npy')
x_test=np.load('test_data.npy')
y_test=np.load('test_labels.npy')

# Softmax regressor to classify images based on encoding 
classifier_model=Sequential()
classifier_model.add(Dense(units=100,input_dim=x_train.shape[1],kernel_initializer='glorot_uniform'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.3))
classifier_model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.2))
classifier_model.add(Dense(units=6,kernel_initializer='he_uniform'))
classifier_model.add(Activation('softmax'))
classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='nadam',metrics=['accuracy'])

classifier_model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))

# Save model for later use
tf.keras.models.save_model(classifier_model,'./face_classifier_model.h5')

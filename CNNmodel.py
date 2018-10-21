import tensorflow as tf
import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

#global variables 
class_names = ["Mayweed", "Charlock"]
paths = ['train/Mayweed/','train/Charlock/','validate/Mayweed/', 'validate/Charlock/']
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 20
EPOCHS = 20

#returns array of images from the specified path, each image is associated with the given label (0,1) or (1,0)
def load_images(input_path, label):
    imgs = []
    for i in os.listdir(input_path):
        img = cv2.imread(input_path+i, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            imgs.append([np.array(img), np.array(label)])
    return imgs

#User interface
if input("Welcome to this simple CNN.\nWould you like to use the preloaded directory tree? (y/n)\n")=="n":
  class_names[0] = input("Please enter the name of the first label\n")
  class_names[1] = input("Please enter the name of the other label\n")
  paths[0] = input("Please enter the path to the training data for "+class_names[0]+"\n")
  paths[1] = input("Please enter the path to the training data for "+class_names[1]+"\n")
  paths[2] = input("Please enter the path to the validation data for "+class_names[0]+"\n")
  paths[3] = input("Please enter the path to the validation data for "+class_names[1]+"\n")
if input("Would you like to use the default number of epochs? (y/n)\n")=="n":
  EPOCHS = input("Please enter the number of epochs to use.\n")

#model is constructed
model = Sequential()
model.add(InputLayer(input_shape=[IMG_HEIGHT,IMG_WIDTH,1]))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, padding='same'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#model is trained
training_set = load_images(paths[0], [1,0])+load_images(paths[1], [0,1])
training_images = np.array([i[0] for i in training_set]).reshape(-1,IMG_HEIGHT,IMG_WIDTH,1)
training_labels = np.array([i[1] for i in training_set])
model.fit(x=training_images, y=training_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
model.summary()

#validation, predications are tested
correct = 0
wrong = 0
validation_set = load_images(paths[2], [1,0])+load_images(paths[3], [0,1])
for oth, data in enumerate(validation_set):
    img = data[0].reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    label = data[1][1]
    prediction = model.predict([img])
    print("Prediction: "+class_names[np.argmax(prediction)]+" , actual: "+class_names[label])
    if np.argmax(prediction) == label:
        correct += 1
    else:
        wrong += 1
print("Accuracy: ", correct / (correct+wrong))
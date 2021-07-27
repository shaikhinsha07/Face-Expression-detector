#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt


# In[2]:


num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 150
epochs = 32


# In[3]:


with open("fer2013.csv") as f:
    content = f.readlines()

lines = np.array(content)


# In[4]:


num_of_instances = lines.size
print("number of instances: ",num_of_instances)
print("instance length: ",len(lines[1].split(",")[1].split(" ")))


# In[5]:


x_train, y_train, x_test, y_test = [], [], [], []


# In[6]:


for i in range(1,num_of_instances):
    try:
        emotion, img, usage = lines[i].split(",")
          
        val = img.split(" ")
            
        pixels = np.array(val, 'float32')
        
        emotion = keras.utils.to_categorical(emotion, num_classes)
    
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
      	print("",end="")


# In[7]:


x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255 #normalize inputs between [0, 1]
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[8]:


model = Sequential()

#1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))


# In[9]:


gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)


# In[10]:


model.compile(loss='categorical_crossentropy'
    , optimizer=keras.optimizers.Adam()
    , metrics=['accuracy']
)


# In[11]:


fit = True

if fit == True:
	#model.fit_generator(x_train, y_train, epochs=epochs) #train for all trainset
	model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs) #train for randomly selected one
else:
	model.load_weights('/data/facial_expression_model_weights.h5') #load weights


# In[12]:


#overall evaluation
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])


# In[13]:


def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()


# In[14]:


monitor_testset_results = False

if monitor_testset_results == True:
	#make predictions for test set
	predictions = model.predict(x_test)

	index = 0
	for i in predictions:
		if index < 30 and index >= 20:
			#print(i) #predicted scores
			#print(y_test[index]) #actual scores
			
			testing_img = np.array(x_test[index], 'float32')
			testing_img = testing_img.reshape([48, 48]);
			
			plt.gray()
			plt.imshow(testing_img)
			plt.show()
			
			print(i)
			
			emotion_analysis(i)
			print("----------------------------------------------")
		index = index + 1


# In[42]:


img = image.load_img("laugh.png", grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.imshow(x)
plt.show()


# In[25]:


from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
import sys
import os
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
import glob
import os
import sys
import warnings
from random import sample
import csv
import cv2


# In[43]:


def UploadAction():
    global filename
    filename = filedialog.askopenfilename()
    print('Selected:', filename)


def DisplayAction():
    global filename
    cv2.imshow(filename)




def fep():
    global filename
    MyWindow = Toplevel(root)

    MyWindow.title("Results")
    MyWindow.geometry("700x250")
    MyWindow.configure(background='light blue')
    image = cv2.imread(filename)  # read file
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # arrange format as per keras
    #image = cv2.resize(image, (224, 224))
    #image = np.array(image) / 255
    #image = np.expand_dims(image, axis=0)
    img = image.load_img(image, grayscale=True)
    x = image.img_to_array(image)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    custom = model.predict(image)
    emotion_analysis(custom[0])
    image = np.array(x, 'float32')
    image = image.reshape([48, 48]);

    plt.gray()
    plt.imshow(x)
    y=plt.show()
    print(y)


# In[44]:


from tkinter import *

root=Tk()
root.configure(background='light yellow')
root.title("EXPRESSION DETECTOR")
root.geometry("500x500")
b=Button(root,text='Open Image for Detecting Expression',bg='yellow',command=UploadAction).grid(row=2,column=1)
display=Button(root,text='result',bg='blue',fg='white',command=fep).grid(row=3,column=1)


root.mainloop()


# In[ ]:





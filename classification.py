                     ########## We work on the Google Colab platform ########

######## We've downloaded six classes from the database,
        #with each class containing 1,380 images in the 
        #training folder,200 images in the test folder,
        #and 350 images in the validation set.##################


#Introduction:

#The approach employed for building the Sign Language Recognition System (SLRS) here is known as Transfer Learning. This technique enables us to leverage the knowledge gained by other models on a closely related problem when learning patterns in the data.

#Through the application of transfer learning, we can achieve superior accuracy compared to constructing a Sign Language Recognition System from the ground up.

#In this notebook, we harness the knowledge and weights acquired by ResNet50V2. While we attempted to employ EfficientNet, the results were unsatisfactory.

    ####  1 Import all the essential libraries necessary for data processing and model building.


                     ########## We work on the Google Colab platform ########


#import the libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg
import seaborn as sns
import random
import zipfile
import pathlib

from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam ,RMSprop
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
import datetime
from tensorflow.keras.applications import  MobileNetV2, ResNet50V2
from tensorflow.keras.preprocessing import image

from sklearn.metrics import confusion_matrix

    #### 2 Inspect the data to find out how many directories are there in the folder and in each directory how many files are present.
for dirpath,dirname,filename in os.walk("/content/drive/MyDrive/full_database/data"):
  print(f'There are {len(dirname)} directories and {len(filename)} images in {dirpath}')
             ## output##
       #There are 3 directories and 0 images in /content/drive/MyDrive/full_database/data
       #There are 6 directories and 0 images in /content/drive/MyDrive/full_database/data/train
       #There are 0 directories and 1380 images in /content/drive/MyDrive/full_database/data/train/call
       #There are 0 directories and 1380 images in /content/drive/MyDrive/full_database/data/train/mute
       #There are 0 directories and 1380 images in /content/drive/MyDrive/full_database/data/train/peace
       #There are 0 directories and 1380 images in /content/drive/MyDrive/full_database/data/train/ok
       #There are 0 directories and 1380 images in /content/drive/MyDrive/full_database/data/train/stop
       #There are 0 directories and 1380 images in /content/drive/MyDrive/full_database/data/train/palm
       #There are 6 directories and 0 images in /content/drive/MyDrive/full_database/data/test
       #There are 0 directories and 200 images in /content/drive/MyDrive/full_database/data/test/call
       #There are 0 directories and 200 images in /content/drive/MyDrive/full_database/data/test/mute
       #There are 0 directories and 200 images in /content/drive/MyDrive/full_database/data/test/peace
       #There are 0 directories and 200 images in /content/drive/MyDrive/full_database/data/test/stop
       #There are 0 directories and 200 images in /content/drive/MyDrive/full_database/data/test/palm
       #There are 0 directories and 200 images in /content/drive/MyDrive/full_database/data/test/ok
       #There are 6 directories and 0 images in /content/drive/MyDrive/full_database/data/validation
       #There are 0 directories and 350 images in /content/drive/MyDrive/full_database/data/validation/stop
       #There are 0 directories and 350 images in /content/drive/MyDrive/full_database/data/validation/palm
       #There are 0 directories and 350 images in /content/drive/MyDrive/full_database/data/validation/ok
       #There are 0 directories and 350 images in /content/drive/MyDrive/full_database/data/validation/peace
       #There are 0 directories and 350 images in /content/drive/MyDrive/full_database/data/validation/mute
       #There are 0 directories and 350 images in /content/drive/MyDrive/full_database/data/validation/call

    # 3  In data we are dealing with Multi-Class classification problem where we have 6 classes of Sign Language Gesture. Thus, to store the names for each class below code is required.

data_dir= pathlib.Path("/content/drive/MyDrive/full_database/data/train")
class_names = [i.name for i in data_dir.glob('*')]
class_names
    ###outout###
   #['call', 'mute', 'peace', 'ok', 'stop', 'palm']

    # 4 Define a function to Visualize the data or the image for target class and pick any one random sample from the directory with its Label.

def view_random_image(target_dir,target_class):
  #setup target directory
  target_folder = target_dir+"/"+target_class
  # print(target_folder)
  #get the random image from the target dir and targetclass
  random_image = random.sample(os.listdir(target_folder),1)
  print(random_image[0])

  #read the image and plot
  img = mpimg.imread(target_folder+"/"+random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis('off')

  print(f'Image shape:{img.shape}')

  return img
img =view_random_image(target_dir="/content/drive/MyDrive/full_database/data/train",target_class='ok')

     # 5 With the help of ImageDataGenerator we can apply random transformations on each of training images while our model is still in training process.
     #Here we have done only rescaling of the images that leads to values between the range of 0 and 1(normalization).
     #If we want to apply more of data augmentation techniques like rotation of image, change of width and height, zoom, flipping etc. this should be valid on the training imagws only
     #Here we have applied ImageGenerator class with flow_from_directory

Image_shape = (224, 224)
Batch_size = 128
DROPOUT = 0.4
L1_REG = 0.001
L2_REG = 0.001

train_dir = '/content/drive/MyDrive/full_database/data/train'
val_dir = '/content/drive/MyDrive/full_database/data/validation'

train_data_gen = ImageDataGenerator(rescale=1/255, 
                                    rotation_range=10, 
                                    width_shift_range=0.1, 
                                    height_shift_range=0.1, 
                                    shear_range=0.1, 
                                    zoom_range=0.1, 
                                    horizontal_flip=True,
                                    brightness_range=[0.5, 1.5] 
                                    )
val_data_gen = ImageDataGenerator(rescale=1/255,
                                  rotation_range=10, 
                                  width_shift_range=0.1, 
                                  height_shift_range=0.1, 
                                  shear_range=0.1, 
                                  zoom_range=0.1, 
                                  horizontal_flip=True,
                                  brightness_range=[0.5, 1.5]  
                                  )

train_data = train_data_gen.flow_from_directory(train_dir, 
                                                target_size=Image_shape,
                                                batch_size=Batch_size,
                                                class_mode='categorical')

valid_data = val_data_gen.flow_from_directory(val_dir, 
                                              target_size=Image_shape,
                                              batch_size=Batch_size,
                                              class_mode='categorical')
    ###output###
#Found 9700 images belonging to 5 classes.
#Found 200 images belonging to 5 classes.

   # 6  Callbacks are extra functionality to perform during and after training tracking experiments ,model checkpoint, early stopping before overfitting

def create_tensorboard_callback(dir_name, experiment_name):
  log_dir=dir_name+'/'+experiment_name+'/'+datetime.datetime.now().strftime('%Y%m%d-%H')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f'saving tensorboardcallback:{log_dir}')
  return tensorboard_callback

    # 7 With the help of Functional API we are building our first Tensorflow Transfer Learning Model with the help of ResNet50V2.
    #While working on the model using transfer learning we are using the FeatureExtraction Technique here. That means we are leveraging the weights of the model and adjusting those weights which would be suited for our classification problem.
    #Over here we'll freeze all the leraned patterns in the bottom layers and we'll adjust the weights of top2-3 pretrained layers of the model in accordance with our custom data. This is our Base Line model which provides an accuracy of 84 on validation data.

base_model = ResNet50V2(include_top=False) 
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

inputs = tf.keras.layers.Input(shape=(224, 224, 3),name='input_layer') 
x = base_model(inputs)

x = tf.keras.layers.GlobalAveragePooling2D(name='gloabl_average2D')(x)
x = Dense(128, activation='relu')(x) 
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(6, activation='softmax', name='output_layer')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(loss='categorical_crossentropy',  
              optimizer=RMSprop(lr=0.0001),  
              metrics=['accuracy'])


history = model.fit(train_data, 
                    epochs=50, 
                    steps_per_epoch=len(train_data) // 8, 
                    validation_data=(valid_data),
                    validation_steps=len(valid_data) // 8)

model.save("model.h5")

    # 8 Visualization Plotting the Loss and Accuracy on Training and Validation data.
#Plot the Graph
# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.savefig('/content/drive/MyDrive/full_database/loss.jpg') 

# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.savefig('/content/drive/MyDrive/full_database/Accuracy.jpg')

   # 9 To look for Predictions on image from the model.
  #Below code can be converted to function, so we can utilize the function again and again without writing the whole code


category = {
    0:'call',
    1:'mute',
    2:'ok',
    3:'palm',
    4:'peace',
    5:'stop'
}

img_=image.load_img('/content/drive/MyDrive/full_database/data/validation/mute/fe0731b5-c221-44b3-81d5-16f338eb0ba6.jpg',target_size=(224,224))
img_array = image.img_to_array(img_)
print(img_array)
img_processed = np.expand_dims(img_array,axis=0)
img_processed /= 255

prediction = model.predict(img_processed)
print(prediction)
index = np.argmax(prediction)
print(index)
plt.title("Prediction - {}".format(category[index]))
plt.imshow(img_array)

    # 10 Evaluation Metrics Let's find the other metrics values and those are:- Precision, Recall, F1 score
## this we'll find with the help of classification report and also try to find the confusion matrix for all the classes.

len(os.listdir('/content/drive/MyDrive/full_database/data/test/call'))
filedir = '/content/drive/MyDrive/full_database/data/test/call'
filedir.split('/')[-1]

   #  11 Function to define the accuracy for each class by doing preprocessing of the images in the test data. So all images in the test data converted to tensors having the exact input shape that we have provided to the images trained in the model.

def predict_dir(filedir,model):
  cols=3
  pos=0
  images=[]
  total_images=len(os.listdir(filedir))
  rows=(total_images//cols+1)
  true = filedir.split('/')[-1]

  for i in sorted(os.listdir(filedir)):
    images.append(os.path.join(filedir,i))

  for subplot,imgg in enumerate(images):
    img_ = image.load_img(imgg,target_size=(224,224))
    img_array = image.img_to_array(img_)
    
    img_processed = np.expand_dims(img_array,axis=0)
    img_processed /= 255

    prediction = model_0.predict(img_processed)
    index = np.argmax(prediction)

    pred = category.get(index)
    if pred==true:
      pos+=1
  accu = pos/total_images
  print("Accuracy for {orignal}: {:.2f} ({pos}/{total})".format(accu,pos=pos,total=total_images,orignal=true))

    # 12 Accuracy for each calss in the test directory. How many of the images have been correctly classified

for i in os.listdir('/content/Fruits_Classification/test'):
  # print(i)
  predict_dir(os.path.join('/content/Fruits_Classification/test',i),model)
  
   # 13  Check the accuracy for each label in the test dataset using confusion_matrix heat map Visualization

from tensorflow.keras.preprocessing import image

def labels_confusion_matix(folder):
  mapping ={}
  for i,j in enumerate(sorted(os.listdir(folder))):
    # print(i)
    # print(j)
    
    mapping[j]=i
  files=[]
  real=[]
  predicted=[]

  for i in os.listdir(folder):
    true = os.path.join(folder,i)
    true = true.split('/')[-1]
    # print(true)
    true = (mapping[true])

    for j in os.listdir(os.path.join(folder,i)):
      img_ = image.load_img(os.path.join(folder,i,j), target_size=(224,224))
      img_array = image.img_to_array(img_)

      img_processed = np.expand_dims(img_array,axis=0)
      img_processed /=255

      prediction = model.predict(img_processed)

      index = np.argmax(prediction)

      predicted.append(index)
      real.append(true)
  return real,predicted


def print_confusion_matrix(real,predicted):
  total_output_labels=5
  cmap='turbo'

  cm_plot_labels=[i for i in range(6)]

  cm = confusion_matrix(y_true=real,y_pred=predicted)

  df_cm = pd.DataFrame(cm,cm_plot_labels,cm_plot_labels)

  sns.set(font_scale=1.2)

  plt.figure(figsize=(15,10))

  s=sns.heatmap(df_cm,fmt="d", annot=True,cmap=cmap)

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig('confusion_matrix.png')
  plt.show()
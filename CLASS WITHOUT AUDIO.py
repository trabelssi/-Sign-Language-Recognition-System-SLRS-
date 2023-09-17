import numpy as np
import cv2
from PIL import Image
import pickle
import time
import pyttsx3

#############################################

frameSize = (320, 240)  # CAMERA RESOLUTION
brightness = 180
threshold = 0.75  # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################


# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameSize[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameSize[1])
cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

# IMPORT THE TRAINED MODEL
from keras.models import load_model
model = load_model('model.h5')

# USE GPU FOR INFERENCE
import tensorflow as tf
if tf.test.is_gpu_available():
    print('Using GPU for inference')
    model = tf.keras.models.Sequential([tf.keras.layers.Lambda(lambda x: x, input_shape=( 120, 320, 1)), model])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
else:
    print('Using CPU for inference')

def preprocessing(imgOrignal):
    img = Image.fromarray(imgOrignal)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img.astype('float32') / 255.0 # normalize the pixel values
    img = img.reshape(1, 224, 224, 3) # reshape to match the input shape of the model
    return img

def getClassName(class_index):
    class_mapping = {
        0: 'call',
        1: 'mute',
        2: 'peace',
        3: 'ok',
        4: 'stop'
    }
    return class_mapping.get(class_index, 'Unknown')

# Text-to-speech setup
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# DISPLAY RESULTS FOR EVERY Nth FRAME
displayFreq = 5
frameCount = 0
startTime = time.time()

while True:

    # READ IMAGE
    success, imgOrignal = cap.read()

    # PROCESS IMAGE
    if success:
        frameCount += 1
        if frameCount % displayFreq == 0:
            img = preprocessing(imgOrignal)

            # PREDICT IMAGE
            predictions = model.predict(img.astype('float32'))
            classIndex = np.argmax(predictions)
            probabilityValue = np.amax(predictions)
            className = getClassName(classIndex)

            # DISPLAY RESULTS
            cv2.putText(imgOrignal, "CLASS: " + className, (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOrignal, "PROBABILITY: " + str(round(probabilityValue * 100, 2)) + "%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Result", imgOrignal)

            
          
               

    # QUIT ON 'Q' KEY
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # PRINT FPS
    if frameCount % 10 == 0:
        fps = frameCount / (time.time() - startTime)
        print('FPS:', fps)

# RELEASE CAMERA AND CLOSE WINDOWS
cap.release()
cv2.destroyAllWindows()

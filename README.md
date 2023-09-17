# Sign Language Recognition System (SLRS)

## Goal

Implement a Convolutional Neural Network in TensorFlow to recognize sign language gestures.

## Incremental Development

Develop the model incrementally, starting with a few gestures and building upwards.

## Results

Final model trained to classify 6 gestures with 92.47% accuracy.

## Dataset

Used the "HaGRID - HAnd Gesture Recognition Image Dataset" dataset from hagrid.

## Usage

1n this project, there are three files:

#classification.py:

This file contains the code that implements the Sign Language Gesture Recognition System (SLRS). It provides all the details about how the system works, including data preprocessing, model training (using the model.h5 file), and making predictions.
The source of the model.h5 file is also provided in this script.

#CLASS WITH AUDIO.py:

This file contains the code for testing the model.h5 on a Raspberry Pi or a laptop. It includes audio-related functionality, likely for incorporating audio input or feedback into the gesture recognition process.

#CLASS WITHOUT AUDIO.py:

This file is similar to the "CLASS WITH AUDIO" file but doesn't include audio-related functionality. It is designed to be faster and may be suitable for situations where audio input or feedback is not necessary or desired.

These three files appear to be part of a project that focuses on Sign Language Gesture Recognition, offering options for both audio-enhanced and audio-free testing of the trained model. You can choose the appropriate file based on your hardware setup and requirements


## Acknowledgments

Thanks to the Deaf and Hard of Hearing communities for their support.



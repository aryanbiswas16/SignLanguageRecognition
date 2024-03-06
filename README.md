# American Sign Language Recognition Bot

Welcome to the repository for our innovative American Sign Language (ASL) Recognition Bot. Utilizing state-of-the-art technology such as MediaPipe, TensorFlow, Keras, OpenCV and SciKit Learn this project aims to bridge the communication gap for those who rely on sign language. This project was my first attempt at applying my knowledge in AI.

## Overview

This ASL Recognition Bot leverages computer vision and machine learning to interpret sign language in real-time. It's a powerful tool that can assist in understanding and translating ASL with ease.

## Features

- Real-time ASL recognition
- High accuracy through deep learning
- Easy-to-use with any webcam

## How It Works

The project consists of several Python scripts, each responsible for a part of the recognition process:

- `DataCollection.py`: Captures the wireframe data of hand gestures using OpenCV and MediaPipe.
- `DataFormat.py`: Formats the collected data, preparing it for model training by separating it into training, testing, and validation sets for use with SciKit Learn.
- `LiveSignLanguageRecognition.py`: The final product that employs OpenCV video capture to read and recognize ASL in real-time.
- `Training.py`: Where the magic happens - the formatted data is fed into a Keras Sequential model to train the recognition system.

## Model Details

The `asl_model.h5` file contains the trained Keras model which powers the recognition engine. It was created using a Sequential model architecture for efficient performance and accurate results.

## Getting Started

1. Clone this repository to your local machine.
2. Ensure you have the prerequisites installed: OpenCV, MediaPipe, TensorFlow, Keras, and SciKit Learn.
3. Run `Training.py` to train the model with your data (see the section on data formatting and collection).
4. Once the model is trained, `LiveSignLanguageRecognition.py` can be used to start recognizing ASL in real-time.

## Prerequisites

To run this project, you'll need:

- Python 3.8 or later
- MediaPipe
- TensorFlow 2.x
- Keras
- OpenCV 4.x
- SciKit Learn

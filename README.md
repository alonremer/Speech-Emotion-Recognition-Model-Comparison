# Speech Emotion Recognition - Comparison
This project compares the performance of two approaches, speech model using embeddings as **Wav2Vec2** and spectrogram-based approach using model as **ResNet18**, for speech emotion recognition using the CREMA-D dataset. The goal is to evaluate the effectiveness of both models in classifying different emotions from audio recordings.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models](#models)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Speech emotion recognition (SER) is a challenging task that aims to classify emotions from speech signals. This project explores and compares two different approaches:
- **Wav2Vec2**: A transformer-based model pre-trained on large speech data and fine-tuned for emotion recognition.
- **ResNet18**: A convolutional neural network typically used for image classification, adapted here for audio-based emotion recognition.

The project uses the CREMA-D dataset, which contains audio recordings of actors expressing various emotions.

## Dataset

The **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset) contains audio and visual recordings of actors reading scripts with different emotional expressions. For this project, only the audio data is used.

- **Classes**: Neutral, Happy, Angry, Sad, Fear, Disgust
- **Number of Samples**: 7,442 audio clips

Due to the dataset's size, it is not included in this repository. You can download it from [this link](https://github.com/CheyneyComputerScience/CREMA-D).

## Models

### Wav2Vec2
Wav2Vec2 is a pre-trained transformer model from the Hugging Face library. It is fine-tuned on the CREMA-D dataset for emotion recognition.

### ResNet18
ResNet18 is a deep convolutional neural network typically used for image classification. Here, we use a modified version for speech emotion recognition by converting audio data into spectrograms.

## Requirements

The project is built with Python 3.8 and requires the following libraries:

```bash
pip install -r requirements.txt

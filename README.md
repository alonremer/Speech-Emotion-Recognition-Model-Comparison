# Speech Emotion Recognition - Model Comparison

<img src="https://github.com/user-attachments/assets/5c34b467-437e-4947-a9d5-b9b24a902d18" alt="SER Comparison" width="600"/>

This project investigates and compares the performance of two distinct approaches to speech emotion recognition using the **CREMA-D** dataset: the **Wav2Vec2** model, which leverages embeddings, and a spectrogram-based approach using **ResNet18**. The primary objective is to evaluate the effectiveness of each model in accurately classifying emotions from audio recordings.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
  
## Project Overview

Speech Emotion Recognition (SER) is a complex task focused on identifying emotions from speech signals. This project explores two different methodologies to tackle this challenge:

- **Wav2Vec2**: A transformer-based model pre-trained on vast amounts of speech data, fine-tuned specifically for emotion recognition tasks.
- **ResNet18**: A convolutional neural network (CNN) traditionally used for image classification, adapted in this project to work with audio data by converting it into spectrograms.

The comparison aims to determine which approach is more effective for the classification of emotions from the provided audio samples.

## Dataset

The **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset) is a comprehensive dataset that includes both audio and visual recordings of actors reading various scripts with a range of emotional expressions. This project utilizes only the audio component of the dataset.

- **Emotion Classes**: Neutral, Happy, Angry, Sad, Fear, Disgust
- **Total Samples**: 7,442 audio clips

Given the dataset's large size, it is not included in this repository. You can download the dataset from [this link](https://github.com/CheyneyComputerScience/CREMA-D).

## Models

### Wav2Vec2
**Wav2Vec2** is a state-of-the-art pre-trained transformer model available through the Hugging Face library. It has been fine-tuned specifically on the CREMA-D dataset for the purpose of emotion recognition.

### ResNet18
**ResNet18** is a deep convolutional neural network (CNN) that is typically employed for image classification tasks. In this project, it has been adapted to recognize emotions from speech by converting the audio data into spectrograms, effectively treating the audio as an image classification problem.

## Results

The evaluation reveals that models utilizing embeddings, such as Wav2Vec2, are generally more effective for processing and understanding audio data compared to converting the data into spectrograms and employing image-based models.

### Wav2Vec2 Performance

![Wav2Vec2 Results](https://github.com/user-attachments/assets/7691216e-6c1c-4269-acc9-4af4801af1a7)

### ResNet18 Performance

<p float="left">
    <img src="https://github.com/user-attachments/assets/de4fd2fe-ca87-44f7-95de-925b976cca7b" width="600" />
    <img src="https://github.com/user-attachments/assets/2553f08b-3d2f-42a5-b38a-ebaefa297800" width="350" />
</p>


## Future Work

Future work could explore the following enhancements:

- **Advanced Models**: Investigate more sophisticated models like Wav2Vec2-BERT, which could potentially yield better performance in emotion recognition tasks.
- **Multi-Dataset Training**: Incorporate additional datasets that feature longer audio clips and utilize contextual spoken text, which might improve the model’s prediction accuracy.
- **Multimodal Fusion**: Integrate video data, allowing the model to utilize facial expressions alongside speech to make more confident emotion predictions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your suggestions or improvements.

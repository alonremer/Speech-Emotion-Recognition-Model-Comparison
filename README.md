# Speech Emotion Recognition - Model Comparison

<img src="https://github.com/user-attachments/assets/5c34b467-437e-4947-a9d5-b9b24a902d18" alt="SER Comparison" width="600"/>

This project investigates and compares the performance of two distinct approaches to speech emotion recognition using the **CREMA-D** dataset: the **Wav2Vec2** model, which leverages embeddings, and a spectrogram-based approach using **ResNet18**. The primary objective is to evaluate the effectiveness of each model in accurately classifying emotions from audio recordings.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Conclusions and Future Work](#conclusions-and-future-work)
- [Contributing](#contributing)
  
## Project Overview

Speech Emotion Recognition (SER) is a complex task focused on identifying and classifying emotional states from spoken language by analyzing vocal features such as tone, pitch, volume, and speech patterns. The goal of SER is to understand and interpret emotions expressed through speech, which has numerous potential applications across various fields, such as mental health and customer service.

This project explores two different methodologies to tackle the challenge of SER:

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
    <img src="https://github.com/user-attachments/assets/de4fd2fe-ca87-44f7-95de-925b976cca7b" width="500" />
    <img src="https://github.com/user-attachments/assets/2553f08b-3d2f-42a5-b38a-ebaefa297800" width="315" />
</p>


## Conclusions and Future Work

Training on spectrograms required a uniform length for the audio data, which involved either padding or trimming. This preprocessing step could potentially hinder the model’s ability to accurately interpret the data. Consequently, utilizing models based on embeddings, like Wav2Vec2, is found to be a more suitable approach for processing audio data compared to converting it into spectrograms and treating the data as images.

Moreover, the short duration of the current audio clips (approximately 2 seconds) limits the available information for emotion recognition. Longer audio clips could provide more context and enhance the model's performance. The current sentences also lack contextual relevance to the emotions being expressed, further constraining the models' effectiveness.

**Future Work:**

Exploring more advanced models, such as Wav2Vec2-BERT, could offer better performance in emotion recognition tasks. Incorporating multiple datasets that feature longer audio clips and potentially utilizing the spoken text could improve prediction accuracy. Additionally, integrating video data, if available, could provide additional context through facial expressions, enhancing the model’s ability to make more confident decisions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your suggestions or improvements.

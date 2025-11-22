This project is a real-time audio classification system built with PyTorch. It uses a Convolutional Recurrent Neural Network (CRNN) to classify live audio from a microphone into different categories such as "Music", "Speech", and "Noise/Silence".

The model is a **Convolutional Recurrent Neural Network (CRNN)** composed of three main parts:

1.  **Convolutional Neural Network**: A series of three convolutional blocks process the input audio spectrogram to extract hierarchical features.
2.  **Recurrent Neural Network**: A two-layer LSTM analyzes the sequence of features over time, capturing temporal dependencies in the audio.
3.  **Classifier**: A final fully connected layer maps the LSTM output to the classification scores for each class.

**Demo**

https://github.com/user-attachments/assets/3e5d97d4-ad89-49b9-a799-8cb67409d072


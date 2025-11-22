import sounddevice as sd
import numpy as np
import threading
import torch
import torchaudio
from model import CRNN
import matplotlib.pyplot as plt

DEVICE_NAME = "Microphone (2- USB Camera-B4.09.24.1), Windows DirectSound" 
DURATION  = 3  # seconds of audio data per inference
CHANNELS = 4
SAMPLE_RATE = 16000
BLOCK_SIZE = (DURATION * SAMPLE_RATE)
SMOOTHING_FACTOR = 0.1
MODEL_NAME = "best_crnn_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# An event flag to signal when new audio data is available
new_data_event = threading.Event()
# A shared variable to hold the latest audio block. Initialize it with zeros.
latest_audio_block = np.zeros((BLOCK_SIZE, CHANNELS), dtype='float32')

def audio_callback(indata, frames, time, status):
    global latest_audio_block
    if status:
        print(status)
    
    # Copy the new data to the shared variable
    latest_audio_block[:] = indata
    # Signal that new data is available
    new_data_event.set()


stream = None

model = CRNN(num_classes=3)
#run on gpu if available
if torch.cuda.is_available():
    model.cuda()
savedmodel = torch.load(MODEL_NAME, map_location=DEVICE)
model.load_state_dict(savedmodel['model_state_dict'])
model.eval()

smoothed_probs = np.array([0, 0, 0]) # Initialize smoothed probabilities
class_names = ['Music', 'Speech', 'Noise'] 


plt.ion() # Turn on interactive mode
fig, ax = plt.subplots(figsize=(8, 4))

try:
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        device=DEVICE_NAME,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    )
    stream.start()
    print("Starting continuous audio stream...")
    print("Press Ctrl+C to stop.")

    while True:
        new_data_event.wait()
        processing_block = latest_audio_block.copy()
        new_data_event.clear()

        # convert to torch tensor
        processing_block = torch.tensor(processing_block, dtype=torch.float32)
        
        # average across the channels
        processing_block = torch.mean(processing_block, dim=1)
        
        # normalize
        max_amp = torch.max(torch.abs(processing_block))
        if max_amp > 0.005:
            processing_block = processing_block / max_amp
        
        # calculate log mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )(processing_block)
        log_mel_spec = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)(mel_spec)
        
        with torch.no_grad():
            model_input = log_mel_spec.unsqueeze(0).unsqueeze(0).to(DEVICE)
            output = model(model_input)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
        
        # Get the prediction for the most recent run
        current_pred_idx = np.argmax(probabilities)
        current_class = class_names[current_pred_idx]

        # Apply exponential moving average for smoothing
        smoothed_probs = SMOOTHING_FACTOR * probabilities + (1 - SMOOTHING_FACTOR) * smoothed_probs
        smoothed_pred_idx = np.argmax(smoothed_probs)
        smoothed_class = class_names[smoothed_pred_idx]

        # Update the plot
        ax.clear() 
        
        # Display the current (instantaneous) prediction
        ax.text(0.5, 0.65, f"Current: {current_class}", 
                ha='center', va='center', fontsize=24, color='gray')
        
        # Display the smoothed prediction
        ax.text(0.5, 0.35, f"Prediction: {smoothed_class}", 
                ha='center', va='center', fontsize=32, color='green', fontweight='bold')

        # remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylim(0, 1)
        
        # Redraw the canvas
        fig.canvas.draw()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    print("\nStopping stream...")
finally:
    plt.ioff() 
    plt.close() 
    if stream is not None:
        stream.stop()
        stream.close()
        print("Stream stopped and closed.")
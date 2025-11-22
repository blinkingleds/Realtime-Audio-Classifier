import torch
import torch.nn as nn

# Define the CRNN model
# It processes input spectrograms to classify audio into different classes.
# Here, we define a CRNN model with 3 convolutional layers followed by a 2-layer LSTM and a fully connected layer.
class CRNN(nn.Module):
    def __init__(self,num_classes=3):
        super(CRNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )
        
        
        self.rnn_input_size = 64 * 8
        self.hidden_size = 128
        
        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size, 
            hidden_size=self.hidden_size, 
            num_layers=2, 
            batch_first=True,
            dropout=0.3
        )
        
        
        self.fc = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.permute(0, 3, 1, 2)
        
        batch_size, seq_len, channels, features = x.size()
        x = x.reshape(batch_size, seq_len, channels * features)
        
        
        x, _ = self.lstm(x)
        
        x = x[:, -1, :]
        x = self.fc(x)
        
        return x
        
        
        
    
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

class LSTM_layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, batch_first):
        super(LSTM_layer, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        # self.fc = nn.Linear(hidden_size, output_size)
        self.h0 = nn.Parameter(torch.zeros(num_layers, batch_size, self.hidden_size))
        self.c0 = nn.Parameter(torch.zeros(num_layers, batch_size, self.hidden_size))
    
    def forward(self, x):
        out, (ht, _) = self.lstm(x, (self.h0, self.c0))
        # out = self.fc(out[:, -1, :])
        return ht[self.num_layers-1]
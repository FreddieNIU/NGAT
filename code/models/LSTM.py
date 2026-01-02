import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
from layers.LSTM_layer import LSTM_layer
from typing import Optional

class LSTM(torch.nn.Module):
    def __init__(self, in_feature:int, out_feature:int, batch_size:int):
        super(LSTM, self).__init__()
        self.hidden = 8
        self.in_head = 8
        self.out_head = 1
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.lstm = LSTM_layer(input_size=self.in_feature, hidden_size=self.hidden, num_layers=1, batch_size=batch_size, batch_first=True)
        # fc for regression ------------------------------------
        # self.fc = nn.Linear(self.in_feature, self.out_feature)
        # --------------------------------------------------------------

        # output for classification ------------------------------------
        self.output = nn.Linear(self.hidden, self.out_feature)
        self.sigmoid = nn.Sigmoid()
        # --------------------------------------------------------------
    def forward(self, data):
        # x = data.x
        x = data.reshape(1, -1, self.in_feature)
        x = self.lstm(x)
        x = F.dropout(x, 0.2, training=self.training)
        
        # # fc for regression-----------
        # x = self.fc(x)
        # return x, 0
        # # ----------------------------

        # output for classification -----
        x = self.output(x)
        x = self.sigmoid(x)
        x = x.squeeze()
        # x = x.view(-1, 2, self.out_feature)
        # logit = x
        # x = F.softmax(logit, dim=1)
        # x = x[:, 0, :]
        # _, x = torch.max(x, dim=1)
        return x, 0
        # -------------------------------
        
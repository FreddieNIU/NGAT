import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.LSTM_layer import LSTM_layer
from layers.TGC_layer import TGCConv
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TGC(torch.nn.Module):
    def __init__(self, in_feature:int, out_feature:int, num_nodes:int, edge_dim:int, matrix_in:bool, batch_size:int, explicit:bool, device):
        super(TGC, self).__init__()
        self.device = device
        self.explicit = explicit
        self.hidden = 64
        self.dropout = 0.0
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.edge_dim = edge_dim
        self.matrix_in = matrix_in
        self.num_nodes=num_nodes
        if self.matrix_in:      # input node featrue is a matrix, thus, need LSTM layer to get the final hidden state, and convert to a vector
            self.lstm_out_feature = 64
            self.lstm = LSTM_layer(input_size=self.in_feature, hidden_size=self.hidden, num_layers=1, batch_size=batch_size, batch_first=True)
            # self.fc1 = nn.Linear(self.hidden, self.hidden)
        else:
            self.fc = nn.Linear(self.in_feature, self.hidden)
        self.conv2 = TGCConv(self.hidden, 1, num_edge_feat=self.edge_dim, explicit=self.explicit)
        ## for regression ----------------
        # self.output = nn.Linear(2*self.hidden,3)
        ## -------------------------------

        ## for classification ------------
        self.output = nn.Linear(2*self.hidden, 4)
        self.sigmoid = nn.Sigmoid()
        ## --------------------------------

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.matrix_in:
            x = self.lstm(x)
            _x = x
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr, device=device)
        x = torch.cat([_x, x], dim=1)
        x = self.output(x)
        x = self.sigmoid(x)
        return x, torch.tensor([[0],[0]])
    
    

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.GCN_layer import GCNConv
from layers.LSTM_layer import LSTM_layer
from typing import Optional

class LSTM_GCN(torch.nn.Module):
    def __init__(self, in_feature:int, out_feature:int, num_nodes:int, edge_dim:int, matrix_in:bool, batch_size:int):
        super(LSTM_GCN, self).__init__()
        self.hidden = 32
        # self.in_head = 8
        self.out_head = 8
        self.dropout = 0.6
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.edge_dim = edge_dim
        self.matrix_in = matrix_in
        self.num_nodes=num_nodes
        if self.matrix_in:      # input node featrue is a matrix, thus, need LSTM layer to get the final hidden state, and convert to a vector
            self.lstm_out_feature = 8
            self.lstm = LSTM_layer(input_size=self.in_feature, hidden_size=self.hidden, num_layers=1, batch_size=batch_size, batch_first=True)
            # self.fc1 = nn.Linear(self.hidden, self.hidden)
        else:
            self.fc = nn.Linear(self.in_feature, self.hidden)
        self.gcnconv2 = GCNConv(self.hidden, self.hidden)
        self.output = nn.Linear(self.hidden*2, self.out_feature)
        ## for regression ---------------------------
        self.sigmoid = nn.Sigmoid()
        ## ------------------------------------------

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.matrix_in:
            x = self.lstm(x)
            lstm_embed = x
        x = F.dropout(x, p=self.dropout, training=self.training)

        gcn_embed = self.gcnconv2(x, edge_index, edge_attr)
        x = torch.cat([gcn_embed,lstm_embed],dim=-1)
        x = F.elu(x)
        x = self.output(x)
        ## for regression ---------------------------
        x = self.sigmoid(x)
        ## ------------------------------------------
        return x, 0
    
    


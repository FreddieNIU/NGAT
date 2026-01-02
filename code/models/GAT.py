import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.GCN_layer import GCNConv
from layers.LSTM_layer import LSTM_layer
from layers.GAT_layer import GATConv
from layers.NGAT_layer import NGATConv
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GAT(torch.nn.Module):
    def __init__(self, in_feature:int, out_feature:int, num_nodes:int, edge_dim:int, matrix_in:bool, batch_size:int, device):
        super(GAT, self).__init__()
        self.device = device
        self.hidden =24
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
        # self.gcnconv2 = GCNConv(self.hidden, self.hidden)
        # self.conv2 = GATConv(self.hidden, self.hidden,  heads=self.out_head, concat=False,edge_dim=self.edge_dim, dropout=self.dropout, alpha=0.2)
        self.conv2 = NGATConv(self.hidden, self.hidden, num_nodes=self.num_nodes, heads=self.out_head, concat=False, edge_dim=self.edge_dim, dropout=self.dropout, alpha=0.2)
        self.output = nn.Linear(self.hidden, self.out_feature)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.matrix_in:
            x = self.lstm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        (x, atten_weight2) = self.conv2(x, edge_index, edge_attr=edge_attr, device=device, return_attention_weights=True)
        # x = self.gcnconv2(x, edge_index, edge_attr=edge_attr, device=self.device)
        x = F.elu(x)
        x = self.output(x)
        return x, torch.mean(atten_weight2, dim=0) # atten_weight2 has shape [heads, N, N], average it on the dim "heads" to get the averaged attention weight of each node on other nodes. Shape of [N, N]
        # return x, 0
    
    


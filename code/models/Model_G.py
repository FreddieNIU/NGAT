import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from layers.GCN_layer import GCNConv
from layers.LSTM_layer import LSTM_layer
from layers.GAT_layer import GATConv
from layers.NGAT_layer import NGATConv
from typing import Optional

class Model_G(torch.nn.Module):
    def __init__(self, in_feature:int, out_feature:int, num_nodes:int, edge_dim:int, matrix_in:bool, batch_size:int, classification:bool):
        super(Model_G, self).__init__()
        self.hidden = 16
        self.out_head = 8
        self.dropout = 0.4
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.edge_dim = edge_dim
        self.matrix_in = matrix_in
        self.classification = classification
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
        self.attention_fc = nn.Linear(self.hidden, 1)
        self.output = nn.Linear(self.num_nodes * self.hidden, self.out_feature)
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.matrix_in:
            x = self.lstm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        (x, atten_weight2) = self.conv2(x, edge_index, edge_attr, return_attention_weights=True)
        # x = self.gcnconv2(x, edge_index, edge_attr)
        x = F.elu(x)

        ##  Compute attention scores [node, hidden] -> [node, 1]
        attention_score = self.attention_fc(x)

        ## Apply softmax to get attention weights [node, 1] -> [node]
        attention_weights = F.softmax(attention_score, dim=0).squeeze(1)
        
        ## Apply attention weights to the node features [node, hidden]
        attention_applied = x * attention_weights.unsqueeze(1)
        
        ## Flatten the attention-applied tensor back to [1, node * hidden]
        x = attention_applied.view(1,-1)

        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output(x)
        if self.classification:
            x = self.sigmoid(x)
        x = x.squeeze()
        return x, atten_weight2[:, 6, 6][0]
        # return x, 0
    
    


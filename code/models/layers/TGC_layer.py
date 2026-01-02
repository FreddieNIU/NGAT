import torch

import torch.nn as nn
import torch.nn.functional as F


class TGCConv(nn.Module):
    def __init__(self, in_features, out_features, num_edge_feat, explicit=True):
        super(TGCConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_edge_feat = num_edge_feat
        self.explicit = explicit

        if self.explicit:   # explicit modeling
            self.weight = nn.Parameter(torch.DoubleTensor(num_edge_feat, out_features)) 
            self.bias = nn.Parameter(torch.DoubleTensor(out_features))
        else:   # implicit modeling
            self.weight = nn.Parameter(torch.DoubleTensor(2 * in_features + num_edge_feat, out_features)) 
            self.bias = nn.Parameter(torch.DoubleTensor(out_features))

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        self.bias.data.fill_(0)

    def forward(self, x, edge_list, edge_attr, device):
        """
        x.shape = [N, in_features]
        edge_list.shape = [2, num_edge]
        edge_attr.shape = [num_edge, num_edge_feat] 
        """
        # Step 1: Linear Transform
        source, target = edge_list
        ei = x[source]   # shape of [num_edge, in_features]
        ej = x[target]   # shape of [num_edge, in_features]

        # Step 2: Calculate the relation strength
        if self.explicit:
            similarity = torch.matmul(ei, ej.T)   # shape of [num_edge, num_edge]
            # eye = torch.eye(ei.shape[0]) # shape of [num_edge, num_edge]
            # simiilarity = torch.sum(simiilarity * eye, dim=1)   # shape of [1, num_edge]
            relation_importance = self.activation(torch.matmul(edge_attr, self.weight) + self.bias)     # shape of [num_edge, out_feature]
            g = torch.matmul(similarity, relation_importance)           # shape of []
        else:
            g = self.activation(torch.matmul(torch.cat([ei, ej, edge_attr], dim=1), self.weight) + self.bias)       # shape of [num_edge, out_feature]

        # Step 3: Project the attention matrix with adjacency matrix
        N = x.shape[0]       # num_nodes
        attention = torch.ones([N, N], device=device)     # initialize attention matrix with zero, shape of [N, N]
        attention[source, target] = g[:, 0]       # assign the positions where there is an edge the corresponding value

        
        # Step 4: Node aggregation
        h_prime = torch.matmul(attention, x)        # shape of [N, out_features]

        # Calculate the number of non-zero elements in each row of the attention matrix
        non_zero_counts = torch.sum(attention != 0, dim=0)  # shape of [N]

        # Avoid division by zero by replacing zeros with ones
        non_zero_counts[non_zero_counts == 0] = 1

        # Divide each row of h_prime by the corresponding element in non_zero_counts
        h_prime = h_prime / non_zero_counts.unsqueeze(1)  # shape of [N, out_features]

        return h_prime  # shape of [heads, N, out_features]
        

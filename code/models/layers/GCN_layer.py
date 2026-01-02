import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.edge_dim = edge_dim

        self.weight = nn.Parameter(torch.DoubleTensor(in_features, out_features))    
        if bias:
            self.bias = nn.Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, edge_list, device, edge_attr=None):
        """
        x.shape = [N, in_features]
        edge_list.shape = [2, num_edge]
        edge_attr.shape = [num_edge, num_edge_feat] 
        """
        # Step 1: Linear Transform
        h = torch.matmul(x, self.weight)        # shape of [heads, N, out_features]
        N = h.shape[0]       # num_nodes
        source, target = edge_list
        
        # Step 2: Get adjacency matrix
        adj = torch.zeros([N, N], device=device)
        if edge_attr is None:
            adj[source, target] = torch.ones([len(source)])
        else:
            adj[source, target] = edge_attr.reshape(-1)

        h_prime = torch.sparse.mm(adj, h)
        if self.bias is not None:
            return h_prime + self.bias
        else:
            return h_prime
  
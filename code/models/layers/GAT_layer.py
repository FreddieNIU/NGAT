import torch
import torch.nn as nn
import torch.nn.functional as F

class GATConv(nn.Module):
    def __init__(self, in_features, out_features, heads, edge_dim, dropout, alpha, concat=True, bias=True):
        super(GATConv, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.edge_dim = edge_dim
        self.alpha = alpha
        self.concat = concat

        self.weight = nn.Parameter(torch.DoubleTensor(self.heads, in_features, out_features))    
        self.a = nn.Parameter(torch.zeros(size=(self.heads, 2*out_features, 1), dtype=torch.double))
        if bias:
            self.bias = nn.Parameter(torch.DoubleTensor(self.heads, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, edge_list, device, edge_attr=None,  return_attention_weights=False):
        """
        x.shape = [N, in_features]
        edge_list.shape = [2, num_edge]
        edge_attr.shape = [num_edge, num_edge_feat] 
        """
        # Step 1: Linear Transform
        x = F.dropout(x, self.dropout, training=self.training)
        h = torch.matmul(x, self.weight)        # shape of [heads, N, out_features]
        
        # Step 2: Calculate the attention weight
        source, target = edge_list      # source, target are of shape [1, num_edge], [1, num_edge]
        a_input = torch.cat([h[:, source], h[:, target]], dim=2)      # shape of [heads, num_edges, 2*out_features]
        e = F.leaky_relu(torch.matmul(a_input, self.a), negative_slope=self.alpha)       # shape of [heads, num_edges, 1]
        e = torch.mul(e, edge_attr)     # attention weight e element-wise multiplied by the edge_attr. e has shape [heads, num_edges, 1], edge_attr has shape [num_edges, 1]
        
        # Step 3: Project the attention matrix with adjacency matrix
        N = h.shape[1]       # num_nodes
        attention = -1e20*torch.ones([self.heads, N, N], device=device, requires_grad=True)     # initialize attention matrix with zero, shape of [heads, N, N]
        attention[:, source, target] = e[:, :, 0]       # assign the positions where there is an edge the corresponding value
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h = F.dropout(h, self.dropout, training=self.training)
        
        # Step 4: Node aggregation
        h_prime = torch.matmul(attention, h)        # shape of [heads, N, out_features]
        if self.bias is not None:
            h_prime = h_prime + self.bias
        
        if self.concat:
            h_prime = h_prime.permute(1,0,2).reshape(N,-1)
        else:
            h_prime = torch.sum(h_prime, dim=0)
        
        if return_attention_weights:
            return h_prime, attention             # shape of [heads, N, out_features], [[heads, N, N]]
        else:
            return h_prime      # shape of [heads, N, out_features]

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGATConv(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, heads, edge_dim, dropout, alpha, concat=True, bias=True):
        super(NGATConv, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.edge_dim = edge_dim
        self.alpha = alpha
        self.concat = concat
        self.num_nodes = num_nodes

        self.weight = nn.Parameter(torch.DoubleTensor(self.heads, in_features, out_features))  
        self.k_weight = nn.Parameter(torch.DoubleTensor(self.heads, in_features, out_features))  
        self.v_weight = nn.Parameter(torch.DoubleTensor(self.heads, in_features, out_features))  
        # self.weight = nn.Parameter(torch.DoubleTensor(self.heads, self.num_nodes, in_features, out_features))    
        # self.a = nn.Parameter(torch.DoubleTensor(self.heads, self.num_nodes, 2*out_features, 1))
        self.a = nn.Parameter(0.01*torch.zeros(size=(self.heads, self.num_nodes, 2*out_features, 1), dtype=torch.double))
        # self.w = nn.Parameter(torch.DoubleTensor([1, 1]))
        # self.w_s = torch.DoubleTensor([1.0, 0.4])
        self.w = nn.Parameter(torch.DoubleTensor(self.heads, 2*out_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.DoubleTensor(self.heads, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.k_weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.v_weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    # Class method for rescaling the parameter "a"
    def rescale_a(self, min_value=0, max_value=1):
        # Check if any value in the parameter tensor "a" has an absolute value less than 1e-3
        if (torch.abs(self.a) < 1e-3).any():
            with torch.no_grad():  # Ensure no gradient is tracked for the rescaling operation
                # Rescale each head individually
                for head in range(self.a.shape[0]):
                    head_a = self.a[head]  # Extract the parameters for this head
                    a_min = head_a.min()  # Minimum for this head
                    a_max = head_a.max()  # Maximum for this head

                    # Avoid division by zero
                    if a_max > a_min:
                        # Normalize between 0 and 1
                        normalized_a = (head_a - a_min) / (a_max - a_min)
                        # Scale to the desired range [min_value, max_value]
                        scaled_a = normalized_a * (max_value - min_value) + min_value
                        # Reassign the scaled value instead of in-place modification
                        self.a[head] = scaled_a.clone()

                        del scaled_a
        else:
            # If no values are less than 1e-3 in absolute value, return without rescaling
            return self.a

    def forward(self, x, edge_list, device, edge_attr=None, return_attention_weights=False):
        """
        x.shape = [N, in_features]
        edge_list.shape = [2, num_edge]
        edge_attr.shape = [num_edge, num_edge_feat] 
        """
        self.rescale_a(min_value=0, max_value=1)
        ## Step 1: Linear Transform
        h = torch.matmul(x, self.weight)
        k = torch.matmul(x, self.k_weight)
        v = torch.matmul(x, self.v_weight)
        # h = torch.matmul(x.reshape(-1, 1, self.in_features), self.weight).reshape(self.heads, self.num_nodes, -1)        # shape of [heads, N, out_features]
        h = F.dropout(h, self.dropout, training=self.training)
        k = F.dropout(k, self.dropout, training=self.training)
        v = F.dropout(v, self.dropout, training=self.training)

        ## Step 2: Calculate the attention weight
        source, target = edge_list      # source, target are of shape [1, num_edge], [1, num_edge]
        # Gather Features for Source and Target Nodes
        source_feat = h[:, source, :]       # Shape: (head, num_edges, out_features)
        target_feat = k[:, target, :]       # Shape: (head, num_edges, out_features)
        concatenated_features = torch.cat([source_feat, target_feat], dim=-1)  # Shape: (head, num_edges, 2 * out_features)

        a_input = torch.zeros(size = (self.heads, self.num_nodes, len(source), 2*self.out_features), device=device)
        """
        a_input[:, source, torch.arange(len(source)), :]:
            ':' : selects all heads.
            'source' : selects the nodes specified in source tensor.
            'torch.arange(len(source))' : selects the positions corresponding to each edge.
            ':' : selects all features in the last dimension.
            'concatenated_features' : contains the concatenated node features for each edge.
        """
        a_input[:, source, torch.arange(len(source)), :] = concatenated_features
        # for i in range(len(source)):
        #     s = source[i]
        #     t = target[i]
        #     s_feat = h[:, s, :]
        #     t_feat = h[:, t, :]
        #     concated_feat = torch.cat((s_feat, t_feat), dim=-1)
        #     a_input[:, s, i, :] = concated_feat
        e = F.leaky_relu(torch.sum(torch.matmul(a_input, self.a), dim=1), negative_slope=self.alpha)
        e = torch.mul(e, edge_attr)     # attention weight e element-wise multiplied by the edge_attr. e has shape [heads, num_edges, 1], edge_attr has shape [num_edges, 1]
        
        ## Step 3: Project the attention matrix with adjacency matrix
        N = h.shape[1]       # num_nodes
        attention = -1e20*torch.zeros([self.heads, N, N], device=device, requires_grad=True)     # initialize attention matrix with 0, shape of [heads, N, N]
        attention[:, source, target] = e[:, :, 0]       # assign the positions where there is an edge the corresponding value
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # h = F.dropout(h, self.dropout, training=self.training)
        
        ## Step 4: Node aggregation
        h_prime = torch.matmul(attention, v)        # shape of [heads, N, out_features]
        # h_prime = self.w_s[0]*h + self.w_s[1]*h_prime

        ## Step5: Combine self feature with aggregated features 
        h_prime = torch.concatenate([h_prime, v], dim=-1)
        h_prime = torch.matmul(h_prime, self.w)
        
        del a_input
        del e

        if self.bias is not None:
            h_prime = h_prime + self.bias
        
        if self.concat:
            h_prime = h_prime.permute(1,0,2).reshape(N,-1)
        else:
            h_prime = torch.sum(h_prime, dim=0)
        
        if return_attention_weights:
            return h_prime, attention               # shape of [N, out_features], [[heads, N, N]]
        else:
            return h_prime                          # shape of [N, out_features]
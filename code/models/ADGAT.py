from torch_geometric.nn import Sequential, GCNConv
from torch.nn import Linear, ReLU
from layers.ADGAT_layer import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AD_GAT(nn.Module):
    def __init__(self, num_stock, d_market, d_news, d_hidden, hidn_rnn, heads_att, hidn_att, dropout=0, alpha=0.2, t_mix = 1, infer = 1, relation_static = 0):
        super(AD_GAT, self).__init__()
        self.t_mix = t_mix
        self.dropout = dropout
        if  self.t_mix == 0: # concat
            self.GRUs_s = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
            self.GRUs_r = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
        elif self.t_mix == 1: # all_tensor
             self.tensor = Graph_Tensor(num_stock,d_hidden,d_market,d_news)
             self.GRUs_s = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
             self.GRUs_r = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
        self.attentions = [
            Graph_Attention(hidn_rnn, hidn_att, dropout=dropout, alpha=alpha, residual=True, concat=True) for _
            in range(heads_att)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.X2Os = Graph_Linear(num_stock, heads_att * hidn_att  + hidn_rnn , 2, bias = True)
        self.reset_parameters()

        ## for regression ----------------
        # self.linear = nn.Linear(2,3)
        ## -------------------------------

        ## for classification ------------
        self.linear = nn.Linear(2, 4)
        self.sigmoid = nn.Sigmoid()
        ## --------------------------------

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def get_relation(self,x_numerical, x_textual, relation_static = None):
        x_r = self.tensor(x_numerical, x_textual)
        x_r = self.GRUs_r(x_r)
        relation = torch.stack([att.get_relation(x_r, relation_static=relation_static) for att in self.attentions])
        # relation_mean = torch.mean(abs_relation,dim = 1)
        return relation

    def get_gate(self,x_numerical,x_textual):
        x_s = self.tensor(x_numerical, x_textual)
        x_s = self.GRUs_s(x_s)
        gate = torch.stack([att.get_gate(x_s) for att in self.attentions])
        return gate

    def forward(self, data):
        x_market = data.x.transpose(0,1)
        x_news = torch.ones(x_market.shape[0], x_market.shape[1], 1)
        x_news = x_news.to(device=device)
        relation_static =  torch.zeros((x_market.shape[1], x_market.shape[1]))
        # print(relation_static.shape)
        source, target = data.edge_index
        relation_static[source, target] = torch.ones(len(source))
        relation_static = relation_static.to(device)

        ## concat vs tensor
        if self.t_mix == 0:  # concat
            x_s = torch.cat([x_market, x_news], dim=-1)
            x_r = torch.cat([x_market, x_news], dim=-1)
        elif self.t_mix == 1:  # concat
            x_s = self.tensor(x_market, x_news)
            x_r = self.tensor(x_market, x_news)
        #GRUs for extract different sequential embedding for relation/gate inferring.
        #Equivalent to use a single GRU and separate non-linear decoders.
        x_s.to(device)
        x_r.to(device)
        x_r = self.GRUs_r(x_r)
        x_s = self.GRUs_s(x_s)
        x_r = F.dropout(x_r, self.dropout, training=self.training)
        x_s = F.dropout(x_s, self.dropout, training=self.training)
        ##
        x = torch.cat([att(x_s, x_r, relation_static = relation_static) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([x, x_s], dim=1)
        x = F.elu(self.X2Os(x))
        ## for regression ----------
        # output = self.linear(x)
        # return output, 0
        ## -------------------------

        ## for classification ------
        x = self.linear(x)
        x = self.sigmoid(x)
        return x, 0
        ## --------------------------

    
class GCN(nn.Module):
    """
    Simple GCN model
    """
    def __init__(self, num_stock, d_market, d_news, d_hidden, hidn_rnn, heads_att, hidn_att, dropout=0, alpha=0.2, t_mix = 1, infer = 1, relation_static = 0):
        super(GCN, self).__init__()

        self.t_mix = t_mix
        self.dropout = dropout
        if  self.t_mix == 0: # concat
            self.GRUs_s = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
            self.GRUs_r = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
        elif self.t_mix == 1: # all_tensor
             self.tensor = Graph_Tensor(num_stock,d_hidden,d_market,d_news)
             self.GRUs_s = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
             self.GRUs_r = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)

        self.gcn_layer = Sequential('x, edge_index', [
             (GCNConv(hidn_rnn, 64), 'x, edge_index -> x'),
             ReLU(inplace=True),
             Linear(64, 1),
        ])

    def forward(self, x_market, x_news, device, relation_static = None):

        if self.t_mix == 0:  # concat
            x_s = torch.cat([x_market, x_news], dim=-1)
            x_r = torch.cat([x_market, x_news], dim=-1)
        elif self.t_mix == 1:  # concat
            x_s = self.tensor(x_market, x_news)
            x_r = self.tensor(x_market, x_news)

        output = self.gcn_layer(x_s,relation_static)
        return output
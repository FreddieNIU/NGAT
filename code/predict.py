import sys 
sys.path.append("../code/graph_building/") 
sys.path.append("../code/models/")
import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from firmgraph_dataset import FirmRelationGraph
# from GAT_copy import GAT
from GAT import GAT
from ADGAT import AD_GAT
from LSTM import LSTM
from LSTM_GCN import LSTM_GCN
from TGC import TGC
from THGNN import THGNN
from evaluator import Evaluaor
from torch.utils.tensorboard import SummaryWriter
from utils import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--load_path',type=str, default='No',
                    help='path of the model file to be loaded')


ROOT_FOLDER = "../"

graph_params = {
    ##  -------- Graph Params ---------
    "dynamic" : "cooccur",
    ##  -------- Node Params ---------
    "num_node_features" : 8,
    "matrix_node_features" : True,
    "node_normalize" : 'mean',
    "num_Y_features" : 3,
    "Y_type" : "std",
    "classification" : False,
    "graph_level_task" : False,
    "estimation_window" : 21,
    ## -------- Edge Params ----------
    "num_edge_features" : 1,
    "weighted_edge" : True,
    "threshold" : 0,
    ## -------- Dataset Params --------
    "root" : ROOT_FOLDER+"data/graph_sets",
    "name" : "ACL2018"
}


if __name__=="__main__":
    args = parser.parse_args()
    ## Load dataset
    dataset = FirmRelationGraph(
                dynamic=graph_params['dynamic'],
                num_node_features=graph_params['num_node_features'],
                matrix_node_features=graph_params['matrix_node_features'],
                node_normalize=graph_params['node_normalize'],
                num_Y_features=graph_params['num_Y_features'],
                Y_type=graph_params['Y_type'],
                estimation_window=graph_params['estimation_window'],
                num_edge_features=graph_params['num_edge_features'],
                weighted_edge=graph_params['weighted_edge'],
                threshold=graph_params['threshold'],
                classification=graph_params['classification'],
                graph_level_task=graph_params['graph_level_task'],
                root = graph_params['root'],
                name=graph_params['name']
            )
    dev_len, test_len = int(len(dataset)/10), int(len(dataset)/10)
    train_len = len(dataset)-dev_len-test_len

    ## Initialize model / Load model
    if args.load_path == "No":
        model = GAT(
            in_feature=graph_params['num_node_features'],
            out_feature=graph_params['num_Y_features'], 
            num_nodes=dataset[0].x.shape[0],
            edge_dim=graph_params['num_edge_features'], 
            matrix_in=graph_params['matrix_node_features'], 
            batch_size=len(dataset.graphgenerator.valid_firm_list),
            device=device
        )
        model.to(device)
    else:
        model = torch.load(args.load_path)
        model.to(device)
    
    ## Logger 
    timestamp = time.strftime("-%Y%m%d-%H%M%S", time.localtime())  #time stamp
    ## Evaluate on Test
    model.eval()
    pred = []
    label = []
    index = []
    for i in tqdm(range(test_len)):
        data = dataset[train_len+dev_len+i]
        data.to(device)
        out, atten_weight = model(data)
        pred.append(out)
        label.append(data.y)
        index.append(torch.tensor(train_len+dev_len+i))
    
    pred = torch.stack(pred)
    label = torch.stack(label)
    index = torch.stack(index)

    # torch.save(pred, 'pred_std.pt' if graph_params['Y_type'] == 'std' else 'pred_mean.pt')
    # torch.save(label, 'label_std.pt' if graph_params['Y_type'] == 'std' else 'label_mean.pt')
    torch.save(index, 'index_std.pt' if graph_params['Y_type'] == 'std' else 'index_mean.pt')
    # torch.save(dataset.graphgenerator.valid_firm_list, 'valid_firms.pt')
    # torch.save(dataset.graphgenerator.valid_trading_days, 'valid_days.pt')
    
    

    



    

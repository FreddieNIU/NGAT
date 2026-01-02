
from torch_geometric.data import Data, InMemoryDataset
from typing import Callable, Optional
import os.path as osp
import os
from graph_generator import GraphGenerator
import torch
import pickle

import warnings
warnings.filterwarnings('ignore')

class FirmRelationGraph(InMemoryDataset):
    def __init__(self, 
                 dynamic:bool,
                 num_node_features:int,
                 matrix_node_features:bool,
                 node_normalize:str,
                 num_Y_features:int,
                 Y_type:str,
                 estimation_window:int,
                 num_edge_features:int,
                 weighted_edge:bool,
                 threshold:int,
                 classification:bool,
                 graph_level_task:bool,

                 root:str, 
                 name:str, 
                 transform:Optional[Callable] = None, 
                 pre_transform:Optional[Callable] = None,
            ):
        assert name.lower() in ['acl2018', "spnews"]
        self.root = root
        if name.lower() == 'acl2018':
            self.dataset_name = 'ACL2018' 
        else:
            self.dataset_name = 'SPNews' 
        self.save_fp = str(osp.join(        # data save path, like "graph_sets/ACL2018/dynamic-cooccur/threshold-0/weighted_edge-True/edge_feature-1/Y_feature-1_Ytype-r_mean5/node_feature-4_norm-mean_matrix-True_ESwindow-20/"
            root, 
            self.dataset_name,
            "dynamic-"+str(dynamic), 
            "threshold-"+str(threshold),
            "weighted_edge_"+str(weighted_edge),
            "edge_feature-"+str(num_edge_features), 
            "Y_feature-"+str(num_Y_features)+"_Ytype-"+Y_type,
            "node_feature-"+str(num_node_features)+"_norm-"+node_normalize+"_matrix-"+str(matrix_node_features)+"_ESwindow-"+str(estimation_window)))
        self.graphgenerator = GraphGenerator(
            dataset_name=self.dataset_name,
            dynamic=dynamic,
            num_node_features=num_node_features,
            matrix_node_features=matrix_node_features,
            node_normalize=node_normalize,
            num_Y_features=num_Y_features,
            Y_type=Y_type,
            estimation_window=estimation_window,
            num_edge_features=num_edge_features,
            weighted_edge=weighted_edge,
            threshold=threshold,
            classification=classification,
            graph_level_task=graph_level_task
        )
        super(FirmRelationGraph, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
        

    @property
    def raw_paths(self) -> str:
        return osp.join(self.save_fp, 'raw')

    @property    ## ## -----------------------------------------------------------------
    def processed_dir(self) -> str:
        return osp.join(self.save_fp, 'processed')
    
    # @property
    # def raw_file_names(self):    # A list of files in the raw_dir which needs to be found in order to skip the download.
    #     if self.dataset_name == "ACL2018":
    #         return "AAPL.csv"
    #     else:
    #         return "ReturnMatrix.csv"

    @property
    def processed_file_names(self):     # A list of files in the processed_dir which needs to be found in order to skip the processing.
        if self.dataset_name == "ACL2018":
            return ['graph_2014-01-02.pt', 'graph_2015-12-30.pt']
        else:
            return ['graph_2022-10-07.pt', 'graph_2023-10-16.pt']
    
    
    # def download(self):     # Downloads raw data into raw_dir
    #     # Download to `self.raw_dir`.
    #     pass
        
    def process(self):      # Processes raw data and saves it into the processed_dir.
        # print('In Process')
        graphOfDays = self.graphgenerator.build_graph()      # call graph generator to generato graph of each day
        data_list = []
        for day in graphOfDays.keys():
            x = torch.tensor(graphOfDays[day]['X'])
            edge_index = None if graphOfDays[day]['Edge_index_Transpose'] is None else torch.tensor(graphOfDays[day]['Edge_index_Transpose']).t()
            edge_attr = None if graphOfDays[day]['Edge_attr'] is None else torch.tensor(graphOfDays[day]['Edge_attr'])
            y = torch.tensor(graphOfDays[day]['Y'])
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)
            # torch.save(data, os.path.join(self.processed_dir, f'data_{day}.pt'))
        self.save(data_list, self.processed_paths[0])
        
        # with open(self.save_fp+'/graphgenerator.pickle', 'wb') as f:        # record the parameters in a file
        #     pickle.dump(self.graphgenerator, f)


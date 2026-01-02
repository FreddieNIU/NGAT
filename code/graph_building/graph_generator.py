from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np
import pandas as pd
from node_generator import NodeGenerator
from edge_generator import EdgeGenerator
from generator import Generator
from tqdm import tqdm
import os

class GraphGenerator(Generator):
    '''
    GraphGenerator Class

    This class is to build the a graph set
    '''
    def __init__(self, dataset_name, dynamic:str='cooccur', num_node_features:int=4, node_normalize:str="mean", num_edge_features:int=1, num_Y_features:int=3, threshold:int=0, weighted_edge:bool=True, matrix_node_features:bool=False, estimation_window:int=20, Y_type:str="mean", classification:bool="False", graph_level_task:bool="False"):
        super().__init__(dataset_name)
        if self.dataset_name == "ACL2018":
            print(f"Generator for {self.dataset_name} is initialized ...")
            print("Total Firms: ", len(os.listdir(self.tweet_fp))-1, ', Valid Firms: ', len(self.valid_firm_list), ", Incomplete Record Firms: ", len(self.valid_firm_list))
            print(f"{len(self.valid_trading_days)} trading days.  {len(self.valid_natural_days)} natual days.")
        elif self.dataset_name == "SPNews":
            print(f"Generator for {self.dataset_name} is initialized ...")
            print("Total Firms: ", len(os.listdir(self.news_fp))-1, ', Valid Firms: ', len(self.valid_firm_list))
            print(f"{len(self.valid_trading_days)} trading days.  {len(self.valid_natural_days)} natual days.")
        if dynamic not in self._valid_dynamic_method:
            raise ValueError(f"{dynamic}, Invalid value of dynamic, dynamic should be one of {self._valid_dynamic_method}!")
        else:
            self._dynamic = dynamic
         # Edge parameter---------------------------------------------
        self._num_edge_features = num_edge_features
        self._weighted_edge = weighted_edge
        self._threshold = threshold
        # Node parameter---------------------------------------------
        self._num_node_features = num_node_features
        self._matrix_node_features = matrix_node_features       # node feature is a matrix if True, node feature is a vector if False
        self._node_normalize = node_normalize
        self._num_Y_features = num_Y_features
        self._Y_type = Y_type
        self._classification = classification
        self.estimation_window = estimation_window
        self.graph_level_task= graph_level_task
         # Generators-------------------------------------------------
        self.node_generator = NodeGenerator(
            self.dataset_name, 
            num_node_features=self._num_node_features, 
            node_normalize=self._node_normalize, 
            num_Y_features=self._num_Y_features, 
            Y_type=self._Y_type, 
            estimation_window=self.estimation_window, 
            matrix_node_features=self._matrix_node_features,
            classification=self._classification,
            )
        self.edge_generator = EdgeGenerator(
            self.dataset_name,
            num_edge_features=self._num_edge_features,
            weighted_edge=self._weighted_edge,
            threshold=self._threshold,
            dynamic = self._dynamic
            )
        self.prev_day_graph = {
            "edge_index_T": None,
            "edge_attr": None
        }

    def pre_execute(self):
        """
        This function it to pre calculate the prediction targets and graph edges of each day.
        """
        for ticker in self.valid_firm_list:     # calculate target df of each ticker, results saved in node_generator ticker_target_dict
            _ = self.node_generator.generate_target_for_one_ticker(ticker)
        
        if self.dataset_name == "ACL2018":
            for day in self.valid_natural_days:     # calculate edges of each day, results saved in edge_generator edge_of_each_day
                if self._dynamic == self._valid_dynamic_method[0]:
                    edge_dict = self.edge_generator.static_cooccurance_in_day_tweet(self.valid_natural_days[0])
                    self.edge_generator.merge_day_edges(day, [edge_dict,])
                elif self._dynamic == self._valid_dynamic_method[1]:
                    edge_dict = self.edge_generator.rulebased_cooccurance_in_day_tweet(day)
                    self.edge_generator.merge_day_edges(day, [edge_dict,])
                elif self._dynamic == self._valid_dynamic_method[2]:
                    corr_edge_dict = self.edge_generator.rolling_corr_inMDayMean(day, ticker_target_dict=self.node_generator.ticker_target_dic, estimation_window=self.estimation_window)
                    cooccur_edge_dict = self.edge_generator.rulebased_cooccurance_in_day_tweet(day)
                    self.edge_generator.merge_day_edges(day, [corr_edge_dict, cooccur_edge_dict])
                elif self._dynamic in self._valid_dynamic_method[3:]:
                    edge_dict = self.edge_generator.rolling_corr_inMDayMean(day, ticker_target_dict=self.node_generator.ticker_target_dic, estimation_window=self.estimation_window)
                    self.edge_generator.merge_day_edges(day, [edge_dict,])
                else:
                    raise ValueError(f"Invalud value for dynamic: {self._dynamic}")
        elif self.dataset_name == "SPNews":
            for day in self.valid_natural_days:     # calculate edges of each day, results saved in edge_generator edge_of_each_day
                if self._dynamic == self._valid_dynamic_method[0]:
                    edge_dict = self.edge_generator.rulebased_cooccurance_in_day_news(self.valid_natural_days[1])
                    self.edge_generator.merge_day_edges(day, [edge_dict,])
                elif self._dynamic == self._valid_dynamic_method[1]:
                    edge_dict = self.edge_generator.rulebased_cooccurance_in_day_news(day)
                    self.edge_generator.merge_day_edges(day, [edge_dict,])
                elif self._dynamic == self._valid_dynamic_method[2]:
                    corr_edge_dict = self.edge_generator.rolling_corr_inMDayMean(day, ticker_target_dict=self.node_generator.ticker_target_dic, estimation_window=self.estimation_window)
                    cooccur_edge_dict = self.edge_generator.rulebased_cooccurance_in_day_news(day)
                    self.edge_generator.merge_day_edges(day, [corr_edge_dict, cooccur_edge_dict])
                elif self._dynamic in self._valid_dynamic_method[3:]:
                    edge_dict = self.edge_generator.rolling_corr_inMDayMean(day, ticker_target_dict=self.node_generator.ticker_target_dic, estimation_window=self.estimation_window, dynamic=self._dynamic)
                    self.edge_generator.merge_day_edges(day, [edge_dict,])
                else:
                    raise ValueError(f"Invalud value for dynamic: {self._dynamic}")
    def get_prev_day_graph(self):
        """
        This function is to save the graph of previous day. 
        In case there is no edges connection in the current day, we assign the previous day graph to the current day.
        """
        return self.prev_day_graph['edge_index_T'], self.prev_day_graph['edge_attr']
    
    def set_prev_day_graph(self, edge_index_T, edge_attr):
        """
        setter of prev_day_graph
        """
        self.prev_day_graph['edge_index_T'] = edge_index_T
        self.prev_day_graph['edge_attr'] = edge_attr

    def build_graph(self):
        """
        This function is to build the graph based on the provided arguments, including num_node_features, num_edge_features, etc.
        Output: the output will be saved in a dict named "graphOfDay" which will be used in the "process" function in building my torch_geometric dataset
        """
        if self.dataset_name == "ACL2018":
            self.edge_generator.reset_tweets_of_each_day()      # reset memory dicts
        elif self.dataset_name == "SPNews":
            self.edge_generator.reset_news_of_each_day()      # reset memory dicts
        
        self.node_generator.reset_ticker_target_dic()       # reset memory dicts
        self.edge_generator.reset_edge_of_each_day()        # reset memory dicts
        graphOfDays = {}
        self.last_period = {}
        _ = self.pre_execute()
        if self.graph_level_task:
            for day in tqdm(self.valid_trading_days):        
                if day in self.valid_natural_days:
                    num_edges_of_day = len(self.edge_generator.edge_of_each_day[day])       # num of edges of this day if there are text information on this day
                else:
                    num_edges_of_day = 0        # num of edges of this day if there is NO text information on this day
                graphOfDays[day] = {
                            'X' : np.zeros((len(self.valid_firm_list),self.estimation_window,self._num_node_features)) if self._matrix_node_features else np.zeros((len(self.valid_firm_list),self._num_node_features)),  # shape is [num_nodes, num_node_features]
                            'Edge_index_Transpose' : None,  # shape is [num_edges, 2], you should transpose before passing them to the data constructor
                            'Edge_attr' : np.zeros((num_edges_of_day,self._num_edge_features)), # shape is [num_edges, num_edge_features]
                            'Y' : np.zeros((self._num_Y_features, ))  # graph-level targets of shape [*,]
                        }
                for ticker_idx in range(len(self.valid_firm_list)):
                    ticker = self.index_to_ticker(ticker_idx)
                    x = self.node_generator.build_node_features(ticker=ticker, day=day)
                    graphOfDays[day]['X'][ticker_idx] = x       # x for one ticker on this day
                y, last_pe = self.node_generator.build_graph_level_y(day=day)
                graphOfDays[day]['Y'] = y       # y for one ticker on this day
                edge_idx_T, edge_attr = self.edge_generator.build_edge_index_and_attr_of_day(day=day)
                if edge_idx_T is None:      # There is no connection today, we use the graph of the previous day
                    graphOfDays[day]['Edge_index_Transpose'], graphOfDays[day]['Edge_attr'] = self.get_prev_day_graph()
                    # pass
                else:
                    graphOfDays[day]['Edge_index_Transpose'] = edge_idx_T       # edge_idx on this day
                    graphOfDays[day]['Edge_attr'] = edge_attr                   # edge_attr on this day
                    _ = self.set_prev_day_graph(edge_index_T=edge_idx_T, edge_attr=edge_attr)
        else:      
            for day in tqdm(self.valid_trading_days):        
                if day in self.valid_natural_days:
                    num_edges_of_day = len(self.edge_generator.edge_of_each_day[day])       # num of edges of this day if there are text information on this day
                else:
                    num_edges_of_day = 0        # num of edges of this day if there is NO text information on this day
                graphOfDays[day] = {
                            'X' : np.zeros((len(self.valid_firm_list),self.estimation_window,self._num_node_features)) if self._matrix_node_features else np.zeros((len(self.valid_firm_list),self._num_node_features)),  # shape is [num_nodes, num_node_features]
                            'Edge_index_Transpose' : None,  # shape is [num_edges, 2], you should transpose before passing them to the data constructor
                            'Edge_attr' : np.zeros((num_edges_of_day,self._num_edge_features)), # shape is [num_edges, num_edge_features]
                            'Y' : np.zeros((len(self.valid_firm_list), self._num_Y_features))  # node-level targets of shape [num_nodes, *]
                        }
                self.last_period[day] = np.zeros((len(self.valid_firm_list), self._num_Y_features))  # node-level last period status. Have the same shape with Y.
                for ticker_idx in range(len(self.valid_firm_list)):
                    ticker = self.index_to_ticker(ticker_idx)
                    x, y, last_pe = self.node_generator.build_node_features_and_node_level_Y(ticker=ticker, day=day)
                    graphOfDays[day]['X'][ticker_idx] = x       # x for one ticker on this day
                    graphOfDays[day]['Y'][ticker_idx] = y       # y for one ticker on this day
                    self.last_period[day][ticker_idx] = last_pe
                edge_idx_T, edge_attr = self.edge_generator.build_edge_index_and_attr_of_day(day=day)
                if edge_idx_T is None:      # There is no connection today, we use the graph of the previous day
                    graphOfDays[day]['Edge_index_Transpose'], graphOfDays[day]['Edge_attr'] = self.get_prev_day_graph()
                    # pass
                else:
                    graphOfDays[day]['Edge_index_Transpose'] = edge_idx_T       # edge_idx on this day
                    graphOfDays[day]['Edge_attr'] = edge_attr                   # edge_attr on this day
                    _ = self.set_prev_day_graph(edge_index_T=edge_idx_T, edge_attr=edge_attr)
        return graphOfDays
    
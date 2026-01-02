import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from generator import Generator

class NodeGenerator(Generator):
    '''
    NodeGenerator Class
    '''
    def __init__(self, 
                 dataset_name, 
                 num_node_features:int, 
                 node_normalize:str,
                 num_Y_features:int,
                 Y_type:str, 
                 estimation_window:int, 
                 matrix_node_features:bool,
                 classification:bool,
                 ):
        super().__init__(dataset_name)
        self.ticker_target_dic = {}
        self.matrix_node_features = matrix_node_features
        self.classification = classification
        if node_normalize not in self._valid_normalize:
            raise ValueError(f"{node_normalize}, Invalid value of node_normalize, node_normalize should be one of {self._valid_normalize}!")
        else:
            self.node_normalize = node_normalize
        if num_node_features not in self._valid_num_node_features:
            raise ValueError(f"{num_node_features}, Invalid value of num_node_features, num_node_features should be one of {self._valid_num_node_features}!")
        else:
            self.num_node_features = num_node_features
        if num_Y_features not in self._valid_num_Y_features:
            raise ValueError(f"{num_Y_features}, Invalid value of num_Y_features, num_Y_features should be one of {self._valid_num_Y_features}!")
        else:
            self.num_Y_features = num_Y_features
        if Y_type not in self._valid_Y_types:
            raise ValueError(f"{Y_type}, Invalid value of Y_type, Y_type should be one of {self._valid_Y_types}!")
        else:
            self.Y_type = Y_type
        if estimation_window not in self._valid_estimation_windows:
            raise ValueError(f"{estimation_window}, Invalid value of estimation_window, estimation_window should be one of {self._valid_estimation_windows}!")
        else:
            self.estimation_window = estimation_window

    def reset_ticker_target_dic(self):
        self.ticker_target_dic = {}

    def rescale(self, df):
        """
        Instead of normalization, we use a series of values like return to rescale, they are log(C/O), log(H/O), log(L/O). 
        And we normalize Open itself.
        """
        df["log C/O"] = 100 * np.log(df['Close']/df['Open'])
        df["log H/O"] = 100 * np.log(df['High']/df['Open'])
        df["log L/O"] = 100 * np.log(df['Low']/df['Open'])
        ## We normalize Open and Volumn
        cols_to_seperate_norm = ['Open', 'Volume']
        df[cols_to_seperate_norm] = df[cols_to_seperate_norm].apply(lambda x: (x - x.mean()) / x.std())
        return df

    def generate_target_for_one_ticker(self, ticker: str):
        """
        This function takes a ticker name as input, calculate the m-step prediction targets.
        Input: a ticker name, estimation window length
        Output: a dataframe with the prediction targets as columns      
        """
        
        if self.dataset_name=="ACL2018":
            df = pd.read_csv(self.price_fp+ticker+'.csv')
            df['log return'] = 100 * np.log(df['Adj Close']/df['Adj Close'].shift(1))      # Calculate Log return = 100 * r(t+1)/r(t)
        elif self.dataset_name == "SPNews":
            df = pd.read_csv(self.price_fp+ticker+'.csv')
            df['log return'] = 100 * np.log(df['Close']/df['Close'].shift(1))      # Calculate Log return = 100 * r(t+1)/r(t)

        # cols_to_seperate_norm = ['Adj Close', 'Volume']     # These values should use column-wise "mu" and "sigma" when normalization
        # cols_to_combined_norm = ['Open', 'High', 'Low', 'Close']    # These are all prices, they should use the same value of "mu" and "sigma" when normalization
        # if self.node_normalize == 'mean':
        #     df[cols_to_seperate_norm] = df[cols_to_seperate_norm].apply(lambda x: (x - x.mean()) / x.std())
        #     df[cols_to_combined_norm] = df[cols_to_combined_norm].apply(lambda x: (x - df[cols_to_combined_norm].mean().mean()) / df[cols_to_combined_norm].std().mean())
        # elif self.node_normalize == "min-max":
        #     df[cols_to_seperate_norm] = df[cols_to_seperate_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        #     df[cols_to_combined_norm] = df[cols_to_combined_norm].apply(lambda x: (x - df[cols_to_combined_norm].min().mean()) / (df[cols_to_combined_norm].max().mean() - df[cols_to_combined_norm].min().mean()))
        # else:
        #     raise ValueError(f"node_normalize has wrong value {self.node_normalize}!")
        
        df = self.rescale(df)
        begin_idx = df[df['Date']==self.period_start].index[0]
        end_idx = df[df['Date']==self.period_end].index[0]
        for m in self.steps:   
            # for each step length, calculate the r_mean and std of log returns
            df['r_mean'+str(m)] = df['log return'].shift(-m).rolling(window=m).mean()
            df['std'+str(m)] = df['log return'].shift(-m).rolling(window=m).std()
            # df['p_mean'+str(m)] = df['Adj Close'].shift(-m).rolling(window=m).mean()
            

            # # 计算r_mean 和 std的第二种方法
            # df['r_mean'+str(m)] = df['log return'].rolling(window=m+self.estimation_window).mean().shift(-m)
            # df['std'+str(m)] = df['log return'].rolling(window=m+self.estimation_window).std().shift(-m)
            

            # for each step length, also calculate the past 5-day 10-day and 21-day average of log return, these are used in the input (node features)
            df['hist_rmean'+str(m)] = df['log return'].rolling(window=m).mean()
            df['hist_std'+str(m)] = df['log return'].rolling(window=m).std()
            # df['hist_pmean'+str(m)] = df['Adj Close'].rolling(window=m).mean()
        if begin_idx < self.estimation_window:
            raise ValueError(f'Historical data length not enough. There are {begin_idx} days ahead of 2014-01-02!')
        target_df = df.iloc[begin_idx-2*self.estimation_window+1:end_idx]
        target_df.reset_index(drop=True, inplace=True)
        self.ticker_target_dic[ticker] = target_df
        return target_df
    

    def visualize_target(self, ticker_df, target_type="Mean"):
        """
        This function is used to print the target in line figure
        """
        if target_type == "Mean":
            plt.figure(figsize=(21,5))
            plt.plot(ticker_df['r_mean21'], label='21')
            plt.plot(ticker_df['r_mean10'], label='10')
            plt.plot(ticker_df['r_mean5'], label='5')
            plt.title('Log Return Mean in 5/10/21 steps')
            plt.xlabel('Trading Day')
            plt.ylabel('Log Return Mean')
            plt.legend()
            plt.show()
        elif target_type == "Std":
            plt.figure(figsize=(21,5))
            plt.plot(ticker_df['std21'], label='21')
            plt.plot(ticker_df['std10'], label='10')
            plt.plot(ticker_df['std5'], label='5')
            plt.title('Log Return std in 5/10/21 steps')
            plt.xlabel('Trading Day')
            plt.ylabel('Log Return Mean')
            plt.legend()
            plt.show()

    def build_node_features_and_node_level_Y(self, ticker:str, day: str):
        """
        This function is to build the node features according to the estimation window and matrix_node_feature. 
        Return node feature X and node level target Y
        """
        ticker_target_df = self.ticker_target_dic[ticker]
        estimation_end = ticker_target_df.index[ticker_target_df['Date'] == day].item()+1   # datafrme slice need the ending index+1 so that the record of that day be included
        estimation_start = estimation_end - self.estimation_window
        ticker_target_df_of_day = ticker_target_df.iloc[estimation_start:estimation_end]    # shape [estimation_window, 14], from "day-estimation_window" to "day", columns:  ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'log return', 'r_mean5', 'std5', 'r_mean10', 'std10', 'r_mean21', 'std21']
        
        ## node feature X
        if self.num_node_features == self._valid_num_node_features[0]:      # 12 features ['Open', 'log H/O', 'log L/O', 'log C/O', 'Volume', 'log return', 'hist_rmean5', 'hist_std5','hist_rmean10', 'hist_std10','hist_rmean21', 'hist_std21']
            ticker_day_X = ticker_target_df_of_day[['Open', 'log H/O', 'log L/O', 'log C/O', 'Volume', 'log return', 'hist_rmean5', 'hist_std5','hist_rmean10', 'hist_std10','hist_rmean21', 'hist_std21']]
        elif self.num_node_features == self._valid_num_node_features[1]:     # 8 features ['Volume', 'log return', 'hist_rmean5', 'hist_std5','hist_rmean10', 'hist_std10','hist_rmean21', 'hist_std21']
           ticker_day_X = ticker_target_df_of_day[['Volume', 'log return', 'hist_rmean5', 'hist_std5','hist_rmean10', 'hist_std10','hist_rmean21', 'hist_std21']]
        elif self.num_node_features == self._valid_num_node_features[2]:     # 2 features['Volume', 'log return']
            ticker_day_X = ticker_target_df_of_day[['Volume', 'log return']]
        else:
            raise KeyError(f"{self.num_node_features} Invalid value of num_node_features.")

        if self.matrix_node_features:    # node feature as matrix
            if ticker_day_X.isna().any().any():
                raise ValueError(f"There is NaN in node feature matrix, ticker: {ticker}, day: {day}")
        else:                       # node feature as vector
            ticker_day_X = ticker_day_X.iloc[-1]    # get the last row
            if ticker_day_X.isna().any():
                raise ValueError(f"There is NaN in node feature vector, ticker: {ticker}, day: {day}")
        
        ## node level Y
        if self.num_Y_features == self._valid_num_Y_features[0]:     #['r_mean1', 'r_mean5', 'r_mean10', 'r_mean21','std5','std10', 'std21']
            if self.Y_type == self._valid_Y_types[-1]:
                ticker_day_Y = ticker_target_df_of_day[['r_mean1', 'r_mean5', 'r_mean10', 'r_mean21','std5','std10', 'std21']]
                y = ticker_day_Y.iloc[-1]
            else:
                raise ValueError(f"{self.Y_type}, Invalid value of Y_type, Y_type should be one of ['all'] when num_Y_features={self._valid_num_Y_features[0]}!")
        elif self.num_Y_features == self._valid_num_Y_features[1]:   
            if self.Y_type == self._valid_Y_types[0]:        #['r_mean1', 'r_mean5', 'r_mean10', 'r_mean21']
                ticker_day_Y = ticker_target_df_of_day[['r_mean1','r_mean5', 'r_mean10', 'r_mean21',]]
                y = ticker_day_Y.iloc[-1]
            else:
                raise ValueError(f"{self.Y_type}, Invalid value of Y_type, Y_type should be ['mean'] when num_Y_features={self._valid_num_Y_features[1]}!")
        elif self.num_Y_features == self._valid_num_Y_features[2]:   
            if self.Y_type == self._valid_Y_types[0]:        #['r_mean5', 'r_mean10', 'r_mean21']
                ticker_day_Y = ticker_target_df_of_day[['r_mean5', 'r_mean10', 'r_mean21',]]
                y = ticker_day_Y.iloc[-1]
            elif self.Y_type == self._valid_Y_types[1]:       #['std5', 'std10', 'std21']
                ticker_day_Y = ticker_target_df_of_day[['std5', 'std10', 'std21']]
                y = ticker_day_Y.iloc[-1]
            else:
                raise ValueError(f"{self.Y_type}, Invalid value of Y_type, Y_type should be one of ['mean','std'] when num_Y_features={self._valid_num_Y_features[1]}!")
        elif self.num_Y_features == self._valid_num_Y_features[3]:
            if self.Y_type == self._valid_Y_types[2]:        #['r_mean5']
                ticker_day_Y = ticker_target_df_of_day[['r_mean5']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[3]:       #['std5']
                ticker_day_Y = ticker_target_df_of_day[['std5']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[4]:        #['r_mean10']
                ticker_day_Y = ticker_target_df_of_day[['r_mean10']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[5]:       #['std10']
                ticker_day_Y = ticker_target_df_of_day[['std10']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[6]:        #['r_mean21']
                ticker_day_Y = ticker_target_df_of_day[['r_mean21']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[7]:       #['std21']
                ticker_day_Y = ticker_target_df_of_day[['std21']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[8]:        #['r_mean10']
                ticker_day_Y = ticker_target_df_of_day[['r_mean1']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[9]:        #['r_mean10']
                ticker_day_Y = ticker_target_df_of_day[['p_mean1']]
                y = ticker_day_Y.iloc[-1].item()
            else:
                raise ValueError(f"{self.Y_type}, Invalid value of Y_type, Y_type should be one of ['mean5','std5','mean10','std10','mean21','std21'] when num_Y_features={self._valid_num_Y_features[2]}!")
        # transfer target to classification labels if for classification task
        if self.classification:
            y, last_pe = self.to_classification_target(y, ticker_day_Y=ticker_day_Y, Y_type=self.Y_type, num_Y_feat = self.num_Y_features)
            return ticker_day_X, y, last_pe
        else:
            return ticker_day_X, y, None


    def to_classification_target(self, y, ticker_day_Y, Y_type, num_Y_feat):
        last_pe = y.copy()
        if Y_type == "mean":
            if num_Y_feat == 4:
                iter = [("r_mean1", 1), ("r_mean5", 5), ("r_mean10", 10), ("r_mean21", 20)]
            else:
                iter = [("r_mean5", 5), ("r_mean10", 10), ("r_mean21", 20)]
        elif Y_type == "std":
            iter = [("std5", 5), ("std10", 10), ("std21", 20)]
        elif Y_type == "all":
            iter = [("r_mean1", 1), ("r_mean5", 5), ("r_mean10", 10), ("r_mean21", 20)]
        elif Y_type == "mean1":
            iter = [("r_mean1", 1)]
        elif Y_type == "mean5":
            iter = [("r_mean5", 5)]
        elif Y_type == "mean10":
            iter = [("r_mean10", 10)]
        elif Y_type == "mean21":
            iter = [("r_mean21", 20)]
        elif Y_type == "std5":
            iter = [("std5", 5)]
        elif Y_type == "std10":
            iter = [("std10", 10)]
        elif Y_type == "std21":
            iter = [("std21", 20)]
        elif Y_type == "p_mean1":
            iter = [("p_mean1", 1)]
        if len(iter) > 1:
            for col, lagging in iter:
                y[col] = 1 if ticker_day_Y[col].iloc[-1] > ticker_day_Y[col].iloc[-1-lagging] else 0
                last_pe[col] = 1 if ticker_day_Y[col].iloc[-1-lagging]>0 else 0
        else:
            for col, lagging in iter:
                y = 1 if ticker_day_Y[col].iloc[-1] > ticker_day_Y[col].iloc[-1-lagging] else 0
                last_pe = 1 if ticker_day_Y[col].iloc[-1-lagging]>0 else 0

        return y, last_pe
    
    def build_node_features(self, ticker:str, day: str):
        """
        This function is to build the node features according to the estimation window and matrix_node_feature. 
        Return node feature X
        """
        ticker_target_df = self.ticker_target_dic[ticker]
        estimation_end = ticker_target_df.index[ticker_target_df['Date'] == day].item()+1   # datafrme slice need the ending index+1 so that the record of that day be included
        estimation_start = estimation_end - self.estimation_window
        ticker_target_df_of_day = ticker_target_df.iloc[estimation_start:estimation_end]    # shape [estimation_window, 14], from "day-estimation_window" to "day", columns:  ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'log return', 'r_mean5', 'std5', 'r_mean10', 'std10', 'r_mean21', 'std21']
        
        ## node feature X
        if self.num_node_features == self._valid_num_node_features[0]:      # 12 features ['Open', 'log H/O', 'log L/O', 'log C/O', 'Volume', 'log return', 'hist_rmean5', 'hist_std5','hist_rmean10', 'hist_std10','hist_rmean21', 'hist_std21']
            ticker_day_X = ticker_target_df_of_day[['Open', 'log H/O', 'log L/O', 'log C/O', 'Volume', 'log return', 'hist_rmean5', 'hist_std5','hist_rmean10', 'hist_std10','hist_rmean21', 'hist_std21']]
        elif self.num_node_features == self._valid_num_node_features[1]:     # 8 features ['Volume', 'log return', 'hist_rmean5', 'hist_std5','hist_rmean10', 'hist_std10','hist_rmean21', 'hist_std21']
           ticker_day_X = ticker_target_df_of_day[['Volume', 'log return', 'hist_rmean5', 'hist_std5','hist_rmean10', 'hist_std10','hist_rmean21', 'hist_std21']]
        elif self.num_node_features == self._valid_num_node_features[2]:     # 2 features['Volume', 'log return']
            ticker_day_X = ticker_target_df_of_day[['Volume', 'log return']]
        else:
            raise KeyError(f"{self.num_node_features} Invalid value of num_node_features.")

        if self.matrix_node_features:    # node feature as matrix
            if ticker_day_X.isna().any().any():
                raise ValueError(f"There is NaN in node feature matrix, ticker: {ticker}, day: {day}")
        else:                       # node feature as vector
            ticker_day_X = ticker_day_X.iloc[-1]    # get the last row
            if ticker_day_X.isna().any():
                raise ValueError(f"There is NaN in node feature vector, ticker: {ticker}, day: {day}")
        
        return ticker_day_X

    def build_graph_level_X(self, day: str):
        """
        This function is to build the node features according to the estimation window and matrix_node_feature. 
        Return node feature X
        """
        df = pd.read_csv(self.dataset_fp+'sp500index.csv')
        
        ## Generate target for index price
        df['log return'] = 100 * np.log(df['Adj Close']/df['Adj Close'].shift(1))      # Calculate Log return = 100 * r(t+1)/r(t)   
        df = self.rescale(df)
        begin_idx = df[df['Date']==self.period_start].index[0]
        end_idx = df[df['Date']==self.period_end].index[0]
        for m in self.steps:   
            # for each step length, calculate the r_mean and std of log returns
            df['r_mean'+str(m)] = df['log return'].shift(-m).rolling(window=m).mean()
            df['std'+str(m)] = df['log return'].shift(-m).rolling(window=m).std()
            df['p_mean'+str(m)] = df['Adj Close'].shift(-m).rolling(window=m).mean()

            # for each step length, also calculate the past 5-day 10-day and 21-day average of log return, these are used in the input (node features)
            df['hist_rmean'+str(m)] = df['log return'].rolling(window=m).mean()
            df['hist_std'+str(m)] = df['log return'].rolling(window=m).std()
            df['hist_pmean'+str(m)] = df['Adj Close'].rolling(window=m).mean()
        if begin_idx < self.estimation_window:
            raise ValueError(f'Historical data length not enough. There are {begin_idx} days ahead of 2114-01-02!')
        target_df = df.iloc[begin_idx-2*self.estimation_window+1:end_idx]
        target_df.reset_index(drop=True, inplace=True)
        estimation_end = target_df.index[target_df['Date'] == day].item()+1   # datafrme slice need the ending index+1 so that the record of that day be included
        estimation_start = estimation_end - self.estimation_window
        target_df_of_day = target_df.iloc[estimation_start:estimation_end]    # shape [estimation_window, 14], from "day-estimation_window" to "day", columns:  ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'log return', 'r_mean5', 'std5', 'r_mean10', 'std10', 'r_mean21', 'std21']
          
        ## node feature X
        if self.num_node_features == self._valid_num_node_features[0]:      # 12 features ['Open', 'log H/O', 'log L/O', 'log C/O', 'Volume', 'log return', 'hist_rmean5', 'hist_std5','hist_rmean10', 'hist_std10','hist_rmean21', 'hist_std21']
            day_X = target_df_of_day[['Open', 'log H/O', 'log L/O', 'log C/O', 'Volume', 'log return', 'hist_rmean5', 'hist_std5','hist_rmean10', 'hist_std10','hist_rmean21', 'hist_std21']]
        elif self.num_node_features == self._valid_num_node_features[1]:     # 8 features ['Volume', 'log return', 'hist_rmean5', 'hist_std5','hist_rmean10', 'hist_std10','hist_rmean21', 'hist_std21']
           day_X = target_df_of_day[['Volume', 'log return', 'hist_rmean5', 'hist_std5','hist_rmean10', 'hist_std10','hist_rmean21', 'hist_std21']]
        elif self.num_node_features == self._valid_num_node_features[2]:     # 2 features['Volume', 'log return']
            day_X = target_df_of_day[['Volume', 'log return']]
        else:
            raise KeyError(f"{self.num_node_features} Invalid value of num_node_features.")

        if self.matrix_node_features:    # node feature as matrix
            if day_X.isna().any().any():
                raise ValueError(f"There is NaN in graph feature matrix, day: {day}")
        else:                       # node feature as vector
            day_X = day_X.iloc[-1]    # get the last row
            if day_X.isna().any():
                raise ValueError(f"There is NaN in graph feature vector, day: {day}")
        
        return day_X
    
    def build_graph_level_y(self, day: str):
        """
        This function is to build the graph level Y 
        Return graph level Y
        """ 
        df = pd.read_csv(self.dataset_fp+'sp500index.csv')
        
        ## Generate target for index price
        df['log return'] = 100 * np.log(df['Adj Close']/df['Adj Close'].shift(1))      # Calculate Log return = 100 * r(t+1)/r(t)   
        df = self.rescale(df)
        begin_idx = df[df['Date']==self.period_start].index[0]
        end_idx = df[df['Date']==self.period_end].index[0]
        for m in self.steps:   
            # for each step length, calculate the r_mean and std of log returns
            df['r_mean'+str(m)] = df['log return'].shift(-m).rolling(window=m).mean()
            df['std'+str(m)] = df['log return'].shift(-m).rolling(window=m).std()
            df['p_mean'+str(m)] = df['Adj Close'].shift(-m).rolling(window=m).mean()

            # for each step length, also calculate the past 5-day 10-day and 21-day average of log return, these are used in the input (node features)
            df['hist_rmean'+str(m)] = df['log return'].rolling(window=m).mean()
            df['hist_std'+str(m)] = df['log return'].rolling(window=m).std()
            df['hist_pmean'+str(m)] = df['Adj Close'].rolling(window=m).mean()
        if begin_idx < self.estimation_window:
            raise ValueError(f'Historical data length not enough. There are {begin_idx} days ahead of 2114-01-02!')
        target_df = df.iloc[begin_idx-2*self.estimation_window+1:end_idx]
        target_df.reset_index(drop=True, inplace=True)
        estimation_end = target_df.index[target_df['Date'] == day].item()+1   # datafrme slice need the ending index+1 so that the record of that day be included
        estimation_start = estimation_end - self.estimation_window
        target_df_of_day = target_df.iloc[estimation_start:estimation_end]    # shape [estimation_window, 14], from "day-estimation_window" to "day", columns:  ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'log return', 'r_mean5', 'std5', 'r_mean10', 'std10', 'r_mean21', 'std21']
        self.graph_target_df = target_df

        ## graph-level Y
        if self.num_Y_features == self._valid_num_Y_features[0]:     #['r_mean5', 'std5', 'r_mean10', 'std10', 'r_mean21','std21']
            if self.Y_type == self._valid_Y_types[-1]:
                ticker_day_Y = target_df_of_day[['r_mean5', 'std5', 'r_mean10', 'std10', 'r_mean21','std21']]
                y = ticker_day_Y.iloc[-1]
            else:
                raise ValueError(f"{self.Y_type}, Invalid value of Y_type, Y_type should be one of ['all'] when num_Y_features={self._valid_num_Y_features[0]}!")
        elif self.num_Y_features == self._valid_num_Y_features[1]:   
            if self.Y_type == self._valid_Y_types[0]:        #['r_mean1', 'r_mean5', 'r_mean10', 'r_mean21']
                ticker_day_Y = target_df_of_day[['r_mean1','r_mean5', 'r_mean10', 'r_mean21',]]
                y = ticker_day_Y.iloc[-1]
            else:
                raise ValueError(f"{self.Y_type}, Invalid value of Y_type, Y_type should be ['mean'] when num_Y_features={self._valid_num_Y_features[1]}!")
        elif self.num_Y_features == self._valid_num_Y_features[2]:   
            if self.Y_type == self._valid_Y_types[0]:        #['r_mean5', 'r_mean10', 'r_mean21']
                ticker_day_Y = target_df_of_day[['r_mean5', 'r_mean10', 'r_mean21',]]
                y = ticker_day_Y.iloc[-1]
            elif self.Y_type == self._valid_Y_types[1]:       #['std5', 'std10', 'std21']
                ticker_day_Y = target_df_of_day[['std5', 'std10', 'std21']]
                y = ticker_day_Y.iloc[-1]
            else:
                raise ValueError(f"{self.Y_type}, Invalid value of Y_type, Y_type should be one of ['mean','std'] when num_Y_features={self._valid_num_Y_features[1]}!")
        elif self.num_Y_features == self._valid_num_Y_features[3]:
            if self.Y_type == self._valid_Y_types[2]:        #['r_mean5']
                ticker_day_Y = target_df_of_day[['r_mean5']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[3]:       #['std5']
                ticker_day_Y = target_df_of_day[['std5']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[4]:        #['r_mean10']
                ticker_day_Y = target_df_of_day[['r_mean10']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[5]:       #['std10']
                ticker_day_Y = target_df_of_day[['std10']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[6]:        #['r_mean21']
                ticker_day_Y = target_df_of_day[['r_mean21']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[7]:       #['std21']
                ticker_day_Y = target_df_of_day[['std21']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[8]:        #['r_mean10']
                ticker_day_Y = target_df_of_day[['r_mean1']]
                y = ticker_day_Y.iloc[-1].item()
            elif self.Y_type == self._valid_Y_types[9]:        #['r_mean10']
                ticker_day_Y = target_df_of_day[['p_mean1']]
                y = ticker_day_Y.iloc[-1].item()
            else:
                raise ValueError(f"{self.Y_type}, Invalid value of Y_type, Y_type should be one of ['mean5','std5','mean10','std10','mean21','std21'] when num_Y_features={self._valid_num_Y_features[2]}!")
        # transfer target to classification labels if for classification task
        if self.classification:
            y, last_pe = self.to_classification_target(y, ticker_day_Y=ticker_day_Y, Y_type=self.Y_type, num_Y_feat = self.num_Y_features)
            return y, last_pe
        else: 
            return y, None
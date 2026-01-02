import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import ast
from collections import Counter
ROOT_FOLDER = "../"


class Generator:
    '''
    Generator Class

    This class is is a base class.
    '''
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.steps = [1, 5, 10, 21]  # steps that we forecast ahead of
        if self.dataset_name == "ACL2018":
            self.dataset_fp = ROOT_FOLDER+"data/raw/ACL2018/"
            self.price_fp = self.dataset_fp + "price/raw/"
            self.tweet_fp = self.dataset_fp + "tweet/preprocessed/"
            self.period_start = "2014-01-02"
            self.period_end = "2016-01-04"
            self.valid_firm_list = self.firm_selection() 
            self.valid_trading_days = self.get_trading_day_list()
            self.valid_natural_days = self.get_valid_natural_days()
            # Node parameter---------------------------------------------
            self.num_nodes = 76
            self._valid_num_node_features = [12, 8, 2] 
            self._valid_normalize = ['mean', 'min-max']
            self._valid_num_Y_features = [7,4,3,1]
            self._valid_Y_types = ['mean','std','mean5','std5','mean10','std10','mean21','std21',"mean1","p_mean1", "all"]
            self._valid_estimation_windows = [21, 40, 100, 250]     # the estimation window length, corresponds to 10,20,100,250 trading days
            # edge parameter---------------------------------------------
            self._valid_thresholds = [0,1,2, 0.5]
            self._valid_dynamic_method = ['static', 'cooccur', 'cooccurNcorr', 'corr_rmean5', 'corr_rmean10', 'corr_rmean20', 'corr_std5', 'corr_std10', 'corr_std20']
        elif self.dataset_name == "SPNews":
            self.dataset_fp = ROOT_FOLDER+"data/raw/SPNews/"
            self.price_fp = self.dataset_fp + "price/"
            self.news_fp = self.dataset_fp + "news/preprocessed_tickerwise/"
            self.news_daily = self.dataset_fp + "news/preprocessed_daily_2/"
            self.period_start = "2022-09-20"
            self.period_end = "2024-04-08"
            self.valid_firm_list = self.firm_selection() 
            # self.all_firm_list = self.all_firms()
            self.valid_trading_days = self.get_trading_day_list()
            self.valid_natural_days = self.get_valid_natural_days()
            # Node parameter---------------------------------------------
            self.num_nodes = 268
            self._valid_num_node_features = [12, 8, 2] 
            self._valid_normalize = ['mean', 'min-max']
            self._valid_num_Y_features = [7,4,3,1]
            self._valid_Y_types = ['mean','std','mean5','std5','mean10','std10','mean21','std21',"mean1","p_mean1", "all"]
            self._valid_estimation_windows = [21, 40, 100, 250]     # the estimation window length, corresponds to 10,20,100,250 trading days
            # edge parameter---------------------------------------------
            self._valid_thresholds = [0,1,2, 0.5]
            self._valid_dynamic_method = ['static', 'cooccur', 'cooccurNcorr', 'corr_rmean5', 'corr_rmean10', 'corr_rmean20', 'corr_std5', 'corr_std10', 'corr_std20']
            
            
    def firm_selection(self):
        """
        This function is to find out the firms whoes text data is incomplete, and find out the valid firm list that we will use.
        """
        if self.dataset_name == "ACL2018":
            # For ACL2018 dataset, find out the firms whose tweets data is not from 2014-01 -- 2015-12, return a list of firms
            firms = []
            valid_firms = []
            for ticker_folder in os.listdir(self.tweet_fp):
                if ticker_folder[0] != '.':
                    tweet_list = os.listdir(self.tweet_fp+ticker_folder)
                    if sorted(tweet_list)[0][:7] != '2014-01':
                        # print(ticker_folder, sorted(tweet_list)[0],sorted(tweet_list)[-1])
                        if ticker_folder not in firms:
                            firms.append(ticker_folder)
                    elif sorted(tweet_list)[-1][:7] != '2015-12':
                        # print(ticker_folder, sorted(tweet_list)[0],sorted(tweet_list)[-1])
                        if ticker_folder not in firms:
                            firms.append(ticker_folder)
                    else:
                        valid_firms.append(ticker_folder)
            
        elif self.dataset_name == "SPNews":
            # For SPNews dataset, find out the firms whose news data is from 2022-09 -- 2023-10, return a list of firms
            firms = []
            valid_firms = []
            for ticker_file in os.listdir(self.news_fp):
                if ticker_file.endswith('.csv'):
                    ticker_name = ticker_file.split('_')[0]
                    price_file_name = ticker_name+".csv"
                    if price_file_name in os.listdir(self.price_fp):
                        df = pd.read_csv(self.price_fp+price_file_name)
                        if df.shape[0] != 0:
                            valid_firms.append(ticker_name)     # save ticker names in a list, in capital, like "AAPL"
                        else:
                            firms.append(ticker_name)
            # print(f"Generator for {self.dataset_name} is initialized ...")
            # print("Total Firms: ", len(os.listdir(self.news_fp))-1, ', Valid Firms: ', len(valid_firms))
        return valid_firms
    
    def all_firms(self):
        """
        This function is to return all of firms that appeared in this dataset, including the valid_firm_list and other firms.
        """
        if self.dataset_name == "ACL2018":
            pass
        elif self.dataset_name == "SPNews":
            valid_firms = []
            # include all firms with trading data
            for ticker_file in os.listdir(self.news_fp):
                if ticker_file.endswith('.csv'):
                    ticker_name = ticker_file.split('_')[0]
                    valid_firms.append(ticker_name)
            # include all firms that has been mentioned in the news
            firms = []
            for d in os.listdir(self.news_daily):
                if d.endswith('.csv'):
                    daily_df = pd.read_csv(self.news_daily+d)
                    if daily_df.shape[0] == 0:
                        pass
                    else:
                        daily_df = daily_df.drop_duplicates(subset='Link', keep="first")
                        daily_df.drop(columns=['Publisher', 'Link', 'ProviderPublishTime', 'Type'], inplace=True)
                        daily_df['related_companies'] = daily_df.apply(lambda x: ast.literal_eval(x['Related Company']) if x['Related Company'][0]=="[" else [x['Company']], axis=1)
                        firms_of_today = daily_df.explode('related_companies')['related_companies']
                        firms.extend(firms_of_today)
                        # firms = list(set(firms))
            # Count the occurrences of each firm
            firm_counts = Counter(firms)

            # Filter the names that appear more than N times
            filtered_firms = [name for name in firm_counts if firm_counts[name] > 60]
            filtered_firms.extend(valid_firms)
            filtered_firms = list(set(filtered_firms))

            print("firm counts: ", len(filtered_firms))

        else:
            raise ValueError(f"Invaid value for dataset_name: {self.dataset_name}")
        return filtered_firms

    def get_trading_day_list(self):
        """
        This function is to get the list of trading days of a dataset.
        Output: return the list of trading days in the dataset.
        """
        if self.dataset_name == "ACL2018":
            for f in self.valid_firm_list:   # check whether all price CSV files has the same length during the period
                try:
                    df = pd.read_csv(self.price_fp+f+'.csv')
                    start = df[df['Date']==self.period_start].index[0]
                    end = df[df['Date']==self.period_end].index[0]
                    dates = df.iloc[start:end]['Date']
                    if len(list(dates)) != 504:
                        print("Incomplete price data record: ",f, len(list(dates)))
                except IndexError:
                    print("Incomplete price data record: ", f)
            aapl_df = pd.read_csv(self.price_fp+"AAPL.csv")
            begin_idx = aapl_df[aapl_df['Date']==self.period_start].index[0]
            end_idx = aapl_df[aapl_df['Date']==self.period_end].index[0]
            dates = list(aapl_df.iloc[begin_idx:end_idx]['Date'])
        elif self.dataset_name == "SPNews":
            for f in self.valid_firm_list:   # check whether all price CSV files has the same length during the period
                try:
                    df = pd.read_csv(self.price_fp+f+'.csv')
                    start = df[df['Date']==self.period_start].index[0]
                    end = df[df['Date']==self.period_end].index[0]
                    dates = list(df.iloc[start:end]['Date'])
                    if len(list(dates)) != 388:
                        print("Incomplete price data record: ",f, len(list(dates)))
                        self.valid_firm_list.remove(f)
                    if df.iloc[start:].shape[0] < 409:
                        print("Incomplete price data record: ",f, len(list(dates)))
                        self.valid_firm_list.remove(f)
                except IndexError:
                    print("Incomplete price data record: ", f, ". Removed from valid_firm_list.")
                    self.valid_firm_list.remove(f)
                    print('Valid Firms: ', len(self.valid_firm_list))
        return dates
    
    def get_valid_natural_days(self) -> list:
        """
        There could be text data out of trading days, like during weekends. 
        This function is to find out all of natual days that has text data in our dataset.
        """
        if self.dataset_name == "ACL2018":
            max = 0
            for tk in os.listdir(self.tweet_fp):        # Loop through all tweet folders and find the ticker which maximum number of CSV files
                if tk[0] != ".":
                    num = len(os.listdir(self.tweet_fp+tk))
                    if num > max:
                        max = num       # max = 696 trading days
                        max_tk = tk     # max_tk should be AAPL, 
            days_not_in_max_tk = []
            for tk in os.listdir(self.tweet_fp):        # Loop through again to find the natual days that max_tk has no tweets but other firms have
                if tk[0] != ".":
                    max_tk_day_list = os.listdir(self.tweet_fp+max_tk)
                    for j in os.listdir(self.tweet_fp+tk):
                        if j not in max_tk_day_list and j not in days_not_in_max_tk:
                            days_not_in_max_tk.append(j)    # days_not_in_max_tk = ['2014-08-05', '2014-11-23']
            natual_days_in_max_tk = os.listdir(self.tweet_fp+max_tk)
            natual_days_in_max_tk.extend(days_not_in_max_tk)
            natual_days_in_max_tk.sort()
            return natual_days_in_max_tk
        elif self.dataset_name == "SPNews":
            # a = os.listdir(self.news_daily)
            # natual_days = list(map(lambda x: x.split(".")[0], a))
            # natual_days.sort()
            df = pd.read_csv(ROOT_FOLDER+"data/raw/SPNews/news/preprocessed_tickerwise/AYI_news.csv")
            start = df[df["Date"]=="2022-09-20"].index[0]
            end = df[df["Date"]=="2024-05-22"].index[0]
            natual_days = list(set(df[start:end]['Date']))
            natual_days.sort()
            natual_days=natual_days[:-1]
            
            return natual_days
    def ticker_to_index(self, ticker: str) -> int:
        """
        Given a ticker, return the corresponding index of this ticker. E.g. AAPL -- 6
        """
        return self.valid_firm_list.index(ticker)
    
    def index_to_ticker(self, index: int) -> str:
        """
        Given an index, return the corresponding ticker of this index. E.g. 6 -- AAPL
        """
        return self.valid_firm_list[index]
    
    def get_prev_days(self, day:str, prev_len:int=5, type:str="natual"):
        day_index = self.valid_trading_days.index(day)
        if day_index >= prev_len-1:     # if there are more than prev_len days before the "day"
            prev_trading_days = self.valid_trading_days[day_index - prev_len+1: day_index+1]  
        else:
            prev_trading_days = self.valid_trading_days[ : day_index+1]
        for i in range(len(prev_trading_days)):
            try:
                first_trading_day = self.valid_natural_days.index(prev_trading_days[i])
                break
            except ValueError:
                pass
        for i in reversed(range(len(prev_trading_days))):
            try:
                last_trading_day = self.valid_natural_days.index(prev_trading_days[-1])
                break
            except ValueError:
                pass
        prev_natual_days = self.valid_natural_days[first_trading_day:last_trading_day+1]
        if type == "natual":
            return prev_natual_days
        elif type == "trading":
            return prev_trading_days
        else:
            raise ValueError(f"Wrong Argument Value for 'type'={type}")
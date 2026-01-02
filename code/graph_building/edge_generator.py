import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from generator import Generator
import itertools
from collections import Counter
import ast

class EdgeGenerator(Generator):
    '''
    EdgeGenerator Class

    This class is to build the edges in a graph
    '''
    def __init__(self, dataset_name, num_edge_features:int, weighted_edge:bool, threshold:int, dynamic:str):
        super().__init__(dataset_name)
        self.edge_of_each_day = {}      # dict to save edge of each day. Key: date, value: day_edge_dict
        self._num_edge_features = num_edge_features
        self._weighted_edge = weighted_edge
        self._dynamic = dynamic
        if threshold not in self._valid_thresholds:
            raise ValueError(f"{threshold}, Invalid value of threshold, threshold should be one of {self._valid_thresholds}!")
        else:
            self._threshold = threshold
        if self.dataset_name == "ACL2018":
            self.tweets_of_each_day = {}    # dict to save tweets of each day. Key: date, value: list of tweets
        elif self.dataset_name == "SPNews":
            self.news_of_each_day = {}

    def reset_edge_of_each_day(self):
        self.edge_of_each_day = {}
    
    def reset_tweets_of_each_day(self):
        self.tweets_of_each_day = {}

    def reset_news_of_each_day(self):
        self.news_of_each_day = {}

    def read_ticker_day_tweet(self, ticker: str, day: str):
        """
        This function is for ACL2018. It is to read one tweet file of a ticker on a certain day.
        Input: e.g. ticker=AAPL, day="2014-01-01"
        Output: a list of dicts which has three keys: "text", "created_at", "user_id_str"
        """
        if self.dataset_name != "ACL2018":
            raise KeyError(f"'read_ticker_day_tweet' function is for ACL2018 dataset, current EdgeGenerator is of {self.dataset_name} dataset.")
        else:
            with open(self.tweet_fp + ticker + "/" + day,'r') as f:
                ticker_day_tweets = []
                for line in f:
                    line_dict = json.loads(line)
                    ticker_day_tweets.append(line_dict)
            return ticker_day_tweets  # ticker_day_tweets is a list of dicts which has three keys: "text", "created_at", "user_id_str"
    
    def remove_duplicates(self, dict_list:list) -> list:
        """
        This function is to remove duplicate tweets in a day. It is specifically for ACL2018 dataset.
        """
        seen = set()
        unique_dicts = []
        
        for d in dict_list:
            # Convert the dictionary to a JSON string
            dict_str = json.dumps(d, sort_keys=True)
            
            if dict_str not in seen:
                seen.add(dict_str)
                unique_dicts.append(d)
        
        return unique_dicts

    def read_day_tweets(self, day: str) -> list:
        """
        This function is for ACL2018. It is to read all tweet files on a certain day.
        Input: e.g. day="2014-01-01"
        Output: a list of dicts which has three keys: "text", "created_at", "user_id_str"
        """
        if self.dataset_name != "ACL2018":
            raise KeyError(f"'read_day_tweet' function is for ACL2018 dataset, current EdgeGenerator is of {self.dataset_name} dataset.")
        else:
            day_tweets = []
            ticker_has_tweets = []
            ticker_has_no_tweets = []
            for ticker in self.valid_firm_list:
                try:
                    ticker_day_tweets = self.read_ticker_day_tweet(ticker=ticker, day=day)
                    day_tweets.extend(ticker_day_tweets)
                    ticker_has_tweets.append(ticker)
                except FileNotFoundError:
                    ticker_has_no_tweets.append(ticker)
                    # print("No such file or directory: ",self.tweet_fp + ticker + "/" + day)
            # print(f"Tweets on {day} is loaded. {len(ticker_has_tweets)} has tweets, {len(ticker_has_no_tweets)} has no tweets.")
            day_tweets = self.remove_duplicates(day_tweets)
            self.tweets_of_each_day[day] = day_tweets
            return day_tweets
    

    def read_day_news(self, day:str):
        """
        This function is for ACL2018. It is to read all tweet files on a certain day.
        Input: e.g. day="2022-09-20"
        Output: a DataFrame of all news on the given day
        """
        if self.dataset_name != "SPNews":
            raise KeyError(f"'read_day_news' function is for SPNews dataset, current EdgeGenerator is of {self.dataset_name} dataset.")
        else:
            day_news = pd.read_csv(self.news_daily + day + ".csv")
            day_news = day_news.drop_duplicates(subset='Link', keep="first")
            day_news.drop(columns=['Publisher', 'Link', 'ProviderPublishTime', 'Type'], inplace=True)
            self.news_of_each_day[day] = day_news
            return day_news

    def add_self_edge(self, day_edges_dict:dict) -> dict:
        """
        This function is to add self edges into the day_edges_dict
        """  
        # max_edge_attr = max(day_edges_dict.values()) if len(day_edges_dict)>0 else 1    # maximum weight of all edges on this day. This value is used to assign to all the self-edges
        for firm in self.valid_firm_list:
            if len(day_edges_dict) > 0:     # maximum weight of all edges on this node. This value is used to assign to all the self-edges
                max_edge_attr = 0
                for pair in day_edges_dict:
                    if pair[0] == firm.lower():
                        if day_edges_dict[pair] > max_edge_attr:
                            max_edge_attr = day_edges_dict[pair]
                if max_edge_attr == 0:
                    max_edge_attr = 1
            else:
                max_edge_attr = 1

            self_pair = (firm.lower(),firm.lower())
            if self_pair not in day_edges_dict:
                day_edges_dict[self_pair] = max_edge_attr   # the weight of self_edge = the maximum weight of all edges on this day
            else:
                day_edges_dict[self_pair] += max_edge_attr  # if the self_edge already exist in the graph, the weight of self_edge += the maximum weight of all edges on this day
        return day_edges_dict

    def rulebased_cooccurance_in_day_tweet(self, day: str) -> dict:
        """
        This function if for ACL2018. It is to find the co-occurance of tickers in tweets by rule based method. 
        In tweets, if both "$aapl" and "$goog" appeared in one tweet article, we count the co-occurance once.
        """
        if self.dataset_name != "ACL2018":
            raise KeyError(f"'rulebased_cooccurance_in_day_tweet' function is for ACL2018 dataset, current EdgeGenerator is of {self.dataset_name} dataset.")
        else:
            day_tweets = self.read_day_tweets(day=day)
            day_edges_dict = {}   # dictioary used to store edges, key: firm pairs has co-occurance, value: co-occurance count
            for tweet in day_tweets:   
                tweet_arr = np.array(tweet['text'])
                idx = tweet_arr=='$'    # Find the index of "$" in the tweet
                idx = np.array(pd.Series(idx).shift(1)).astype('bool')  # Find the index of any word following "$", which is usually the ticker of a firm
                idx[0] = False
                firms_mentioned_in_tweet = tweet_arr[idx]   # Find out the firms that has been mentioned in this tweet
                valid_firms_mentioned_in_tweet = []  
                for firm in firms_mentioned_in_tweet:  # Omit the firms that are not in valid_firm_list
                    if firm.upper() in self.valid_firm_list:
                        valid_firms_mentioned_in_tweet.append(firm)
                valid_firms_mentioned_in_tweet = list(set(valid_firms_mentioned_in_tweet))      # 每个公司tag只保留一次
                if len(valid_firms_mentioned_in_tweet) >= 2:  # More than 1 firm mentioned
                    permutations = list(itertools.permutations(valid_firms_mentioned_in_tweet, 2)) #生成所有元素两两分组的排列，包括不同顺序的组合
                    for firm_pair in permutations:
                        if firm_pair not in day_edges_dict:
                            day_edges_dict[firm_pair] = 1
                        else:
                            day_edges_dict[firm_pair] += 1
            day_edges_dict = self.add_self_edge(day_edges_dict)     # add self edges
            return day_edges_dict
        
    def static_cooccurance_in_day_tweet(self, first_day: str)-> dict:
        """
        This function if for ACL2018. It is to find the co-occurance of tickers in tweets by rule based method. 
        In tweets, if both "$aapl" and "$goog" appeared in one tweet article, we count the co-occurance once.
        But we assign the cooccurance in the first day through all period since this function is for static graph set.
        """
        if self.dataset_name != "ACL2018":
            raise KeyError(f"'static_cooccurance_in_day_tweet' function is for ACL2018 dataset, current EdgeGenerator is of {self.dataset_name} dataset.")
        else:
            day_tweets = self.read_day_tweets(day=first_day)
            day_edges_dict = {}   # dictioary used to store edges, key: firm pairs has co-occurance, value: co-occurance count
            for tweet in day_tweets:   
                tweet_arr = np.array(tweet['text'])
                idx = tweet_arr=='$'    # Find the index of "$" in the tweet
                idx = np.array(pd.Series(idx).shift(1)).astype('bool')  # Find the index of any word following "$", which is usually the ticker of a firm
                idx[0] = False
                firms_mentioned_in_tweet = tweet_arr[idx]   # Find out the firms that has been mentioned in this tweet
                valid_firms_mentioned_in_tweet = []  
                for firm in firms_mentioned_in_tweet:  # Omit the firms that are not in valid_firm_list
                    if firm.upper() in self.valid_firm_list:
                        valid_firms_mentioned_in_tweet.append(firm)
                if len(valid_firms_mentioned_in_tweet) >= 2:  # More than 1 firm mentioned
                    permutations = list(itertools.permutations(valid_firms_mentioned_in_tweet, 2)) #生成所有元素两两分组的排列，包括不同顺序的组合
                    for firm_pair in permutations:
                        if firm_pair not in day_edges_dict:
                            day_edges_dict[firm_pair] = 1
                        else:
                            day_edges_dict[firm_pair] += 1
            day_edges_dict = self.add_self_edge(day_edges_dict)     # add self edges
            return day_edges_dict

    def rolling_corr_inMDayMean(self, day: str, ticker_target_dict:dict, estimation_window:int, dynamic:str='corr_rmean20')->dict:
        """
        This function build edges based on the rolling correlation of the historical m day rmean and std.
        If the rolling correlation between two tickers in the past estimation window is larger than a threshold, we build an edge between them.
        """
        if day not in self.valid_trading_days:  # not a trading day, no trading data records. 
            return {}
        else:
            thresholds = {1:0.95, 5:0.9, 10:0.85, 21:0.8}
            day_rmean_edges_dict = {}   # dictioary used to store edges, key: firm pairs has co-occurance, value: co-occurance count
            day_std_edges_dict = {}
            ## Concatenate the columns that used to calculate the correlation  in each ticker df in to one df
            rmeans = {}    # key is step, value is a list of pd.Series
            stds = {}
            concatenated_rmeans = {}    # key is step, value is a pd.DataFrame of concatenated hist_rmean{step}
            concatenated_std = {}
            for step in self.steps: 
                rmeans[step] = []
                stds[step] = []
                day_rmean_edges_dict[step] = {}   #used to save the edges calculated based on the correlation
                day_std_edges_dict[step] = {}
            df = ticker_target_dict['AAL']
            idx = df.index[df['Date']==day].item()      # find the index of the day in ticker df
            for step in self.steps:     # concatenate the hist_rmeans into one df, concatenate the hist_std into one df
                for ticker in self.valid_firm_list:
                    hist_rmean_step = ticker_target_dict[ticker]['hist_rmean'+str(step)]
                    hist_rmean_step.rename(ticker, inplace=True)
                    hist_std_step = ticker_target_dict[ticker]['hist_std'+str(step)]
                    hist_std_step.rename(ticker, inplace=True)
                    rmeans[step].append(hist_rmean_step)
                    stds[step].append(hist_std_step)
            for step in self.steps:         ## Now we have 6 concatenated df that are in concatenated_rmeans and concatenated_std
                # concatenated_rmeans[step] = pd.concat([df for df in rmeans[step]], axis=1)
                # concatenated_rmeans[step] = concatenated_rmeans[step].loc[idx-estimation_window+1 : idx]    # only used data from idx-es to idx to calculate the correlation
                # concatenated_std[step] = pd.concat([df for df in stds[step]], axis=1)
                # concatenated_std[step] = concatenated_std[step].loc[idx-estimation_window+1 : idx]      # only use data from idx-es to idx to calculate the correlation
                # rmean_corr_df = concatenated_rmeans[step].corr()
                # std_corr_df = concatenated_std[step].corr()
                # for source in rmean_corr_df.columns:
                #     # build edge between tickers whose correlation of rmean in the past step days is larger than threshold[step]
                #     target_list = rmean_corr_df[source].index[(rmean_corr_df>thresholds[step])[source]==True]
                    
                #     for target in target_list:
                #         if target != source:
                #             day_rmean_edges_dict[step][(source.lower(), target.lower())] = rmean_corr_df[source].loc[target]
                #     # build edge between tickers whose correlation of std in the past step days is larger than threshold[step]
                #     target_list = std_corr_df[source].index[(std_corr_df>thresholds[step])[source]==True]
                #     for target in target_list:
                #         if target != source:
                #             day_std_edges_dict[step][(source.lower(), target.lower())] = std_corr_df[source].loc[target]
                concatenated_rmeans[step] = pd.concat([df for df in rmeans[step]], axis=1)
                concatenated_std[step] = pd.concat([df for df in stds[step]], axis=1)
                for source in concatenated_rmeans[step].columns:
                    source_ticker_df = concatenated_rmeans[step][source].loc[idx-estimation_window+1 : idx]     # source ticker的数据是从day-es+1到day
                    source_ticker_df.reset_index(drop=True, inplace=True)
                    
                    # other_ticker_df = concatenated_rmeans[step].loc[idx-2*estimation_window+1 : idx-estimation_window]   # 其他ticker的数据是从day-2*es 到day-es+1
                    other_ticker_df = concatenated_rmeans[step].loc[idx-estimation_window+1 : idx]
                    
                    other_ticker_df.drop(columns=[source], inplace=True)
                    other_ticker_df.reset_index(drop=True, inplace=True)
                    concat_df = pd.concat([source_ticker_df, other_ticker_df], axis=1)
                    rmean_corr_df = concat_df.corr()
                    source_ticker_df = concatenated_std[step][source].loc[idx-estimation_window+1 : idx]     # source ticker的数据是从day-es+1到day
                    source_ticker_df.reset_index(drop=True, inplace=True)
                    
                    # other_ticker_df =  concatenated_std[step].loc[idx-2*estimation_window+1 : idx-estimation_window]   # 其他ticker的数据是从day-2*es 到day-es+1
                    other_ticker_df =  concatenated_std[step].loc[idx-estimation_window+1 : idx]
                    
                    other_ticker_df.drop(columns=[source], inplace=True)
                    other_ticker_df.reset_index(drop=True, inplace=True)
                    concat_df = pd.concat([source_ticker_df, other_ticker_df], axis=1)
                    std_corr_df = concat_df.corr()
                    # build edge between tickers whose correlation of rmean in the past step days is larger than threshold[step]
                    target_list = []
                    postive = rmean_corr_df[source].index[(rmean_corr_df>thresholds[step])[source]==True]
                    negative = rmean_corr_df[source].index[(rmean_corr_df<-thresholds[step])[source]==True]
                    target_list.extend(list(postive))
                    target_list.extend(list(negative)) 
                    for target in target_list:
                        if target != source:
                            day_rmean_edges_dict[step][(source.lower(), target.lower())] = rmean_corr_df[source].loc[target]
                    # build edge between tickers whose correlation of std in the past step days is larger than threshold[step]
                    target_list = []
                    postive = std_corr_df[source].index[(std_corr_df>thresholds[step])[source]==True]
                    negative = std_corr_df[source].index[(std_corr_df<-thresholds[step])[source]==True]
                    target_list.extend(list(postive))
                    target_list.extend(list(negative)) 
                    for target in target_list:
                        if target != source:
                            day_std_edges_dict[step][(source.lower(), target.lower())] = std_corr_df[source].loc[target]
                ## Now we have 6 day_edges_dicts saved in day_rmean_edges_dict, day_std_edges_dict
            ## Lets only build edges for tickers has longer corr (in 20 days) only
            # day_edges_dict = dict(Counter(day_rmean_edges_dict[self.steps[2]]) + Counter(day_std_edges_dict[self.steps[2]]))
            if dynamic == self._valid_dynamic_method[3]:    # corr_rmean5
                day_edges_dict = day_rmean_edges_dict[self.steps[1]]
            elif dynamic == self._valid_dynamic_method[4]:    # corr_rmean10
                day_edges_dict = day_rmean_edges_dict[self.steps[2]]
            elif dynamic == self._valid_dynamic_method[5]:    # corr_rmean20
                day_edges_dict = day_rmean_edges_dict[self.steps[3]]
            elif dynamic == self._valid_dynamic_method[6]:    # corr_std5
                day_edges_dict = day_std_edges_dict[self.steps[1]]
            elif dynamic == self._valid_dynamic_method[7]:    # corr_std10
                day_edges_dict = day_std_edges_dict[self.steps[2]]
            elif dynamic == self._valid_dynamic_method[8]:    # corr_std20
                day_edges_dict = day_std_edges_dict[self.steps[3]]
            else:
                raise ValueError(f"Invalud value for dynamic: {dynamic}")
            day_edges_dict = self.add_self_edge(day_edges_dict)     # add self edges
            return day_edges_dict
    
    def rulebased_cooccurance_in_day_news(self, day:str) -> dict:
        """
        This function if for SPNews. It is to find the co-occurance of tickers in news by rule based method. 
        In one row of record of company A, if the Related Company = ['UNH', 'F', 'AAPL'], then we count the cooccurance like follows:
        (A, UNH) = 1
        (A, F) = 1
        (A, AAPL) = 1
        (UNH, F) = 1
        (UNH, AAPL) = 1
        (F, AAPL) = 1
        """
        pass
        if self.dataset_name != "SPNews":
            raise KeyError(f"'rulebased_cooccurance_in_day_news' function is for SPNews dataset, current EdgeGenerator is of {self.dataset_name} dataset.")
        else:
            day_news = self.read_day_news(day=day)
            if day_news.shape[0]==0:
                day_edges_dict={}
            else:
                day_edges_dict = {}   # dictioary used to store edges, key: firm pairs has co-occurance, value: co-occurance count
                try:
                    day_news = day_news[~day_news['Text'].str.startswith("[In this article")]   # Discard those summary news
                except TypeError:
                    day_news = day_news[~day_news['Text'].str.startswith("[In this article").replace(np.nan, True)]
                day_news['related_companies'] = day_news.apply(lambda x: ast.literal_eval(x['Related Company']) if x['Related Company'][0]=="[" else [x['Company']], axis=1)
                # ------------------------------------------------------------
                # # Explode the related_companies list into multiple rows, and drop the rows whose related_comapny is the source company itself. This is to match the logic that we used in rulebased cooccurance in day tweets method.
                # df_exploded = day_news.explode('related_companies')
                # mask = df_exploded['Company'] != df_exploded['related_companies']
                # df_exploded = df_exploded[mask]
                # # df_exploded
                
                # # Generate pairs (source, related) and (related, source)
                # pairs = pd.concat([
                #     df_exploded[['Company', 'related_companies']],
                #     df_exploded[['related_companies', 'Company']].rename(columns={'related_companies': 'Company', 'Company': 'related_companies'})
                # ])
                # # Filter out the ticker that is not in valid_firm_list
                # pairs.reset_index(drop=True, inplace=True)
                # pairs = pairs[pairs['related_companies'].isin(self.valid_firm_list)]
                # pairs.reset_index(drop=True, inplace=True)
                # pairs = pairs[pairs['Company'].isin(self.valid_firm_list)]
                # pairs.reset_index(drop=True, inplace=True)      

                # # Count occurrences of each pair
                # pair_counts = pairs.groupby(['Company', 'related_companies']).size().reset_index(name='count')

                # # Set the index to be a MultiIndex with source and related_companies
                # pair_counts = pair_counts.set_index(['Company', 'related_companies'])

                # # Convert to dictionary
                # day_edges_dict = pair_counts['count'].to_dict()
                # ------------------------------------------------------------

                # day_news['permutations'] = day_news.apply(lambda x: list(itertools.permutations(x['related_companies'], 2)), axis=1)
                day_news['permutations'] = day_news.apply(lambda x: list(itertools.permutations(x['related_companies'], 2)) if len(x['related_companies'])>1 else [(x['related_companies'][0], x['related_companies'][0])], axis=1)


                # Flatten the Series into a list of tuples
                all_pairs = [pair for sublist in day_news["permutations"] for pair in sublist]

                # Count the occurrences of each tuple using Counter
                pair_counts = Counter(all_pairs)

                # Convert the Counter to a dictionary (if needed)
                day_edges_dict = {pair: count for pair, count in pair_counts.items() if pair[0] in self.valid_firm_list and pair[1] in self.valid_firm_list}

                day_edges_dict = self.add_self_edge(day_edges_dict)     # add self edges
            return day_edges_dict

    def static_cooccurance_in_day_news(self, first_day: str)-> dict:
        """
        This function if for SPNews. It is to find the co-occurance of tickers in tweets by rule based method. 
        """
        if self.dataset_name != "SPNews":
            raise KeyError(f"'rulebased_cooccurance_in_day_news' function is for SPNews dataset, current EdgeGenerator is of {self.dataset_name} dataset.")
        else:
            day_news = self.read_day_news(day=first_day)
            if day_news.shape[0]==0:
                day_edges_dict={}
            else:
                day_edges_dict = {}   # dictioary used to store edges, key: firm pairs has co-occurance, value: co-occurance count
                try:
                    day_news = day_news[~day_news['Text'].str.startswith("[In this article")]   # Discard those summary news
                except TypeError:
                    day_news = day_news[~day_news['Text'].str.startswith("[In this article").replace(np.nan, True)]
                day_news['related_companies'] = day_news.apply(lambda x: ast.literal_eval(x['Related Company']) if x['Related Company'][0]=="[" else [x['Company']], axis=1)
                
                day_news['permutations'] = day_news.apply(lambda x: list(itertools.permutations(x['related_companies'], 2)) if len(x['related_companies'])>1 else [(x['related_companies'][0], x['related_companies'][0])], axis=1)

                # Flatten the Series into a list of tuples
                all_pairs = [pair for sublist in day_news["permutations"] for pair in sublist]

                # Count the occurrences of each tuple using Counter
                pair_counts = Counter(all_pairs)

                # Convert the Counter to a dictionary (if needed)
                day_edges_dict = {pair: count for pair, count in pair_counts.items() if pair[0] in self.valid_firm_list and pair[1] in self.valid_firm_list}

                day_edges_dict = self.add_self_edge(day_edges_dict)     # add self edges
            return day_edges_dict
        
    def merge_day_edges(self, day:str, day_edge_dicts:list):
        total = Counter()
        for i in day_edge_dicts:
            total += Counter(i) 
        self.edge_of_each_day[day] = dict(total)

    def weighted_sum_dicts(self, dict_list):
        # Calculate the weights for each dictionary
        n = len(dict_list)
        weights = [(i + 1) / sum(range(1, n + 1)) for i in range(n)]

        # Initialize an empty dictionary to store the weighted sums
        weighted_sum = {}

        # Iterate through the list of dictionaries
        for idx, d in enumerate(dict_list):
            weight = weights[idx]
            for key, value in d.items():
                if key in weighted_sum:
                    weighted_sum[key] += value * weight
                else:
                    weighted_sum[key] = value * weight

        return weighted_sum

    def normalize_attr(self, edge_attr):
        """
        This function is to normalize the edge_attr (np array)
        """
        # x_min = edge_attr.min(axis=0)
        x_max = edge_attr.max(axis=0)
        # x_normalized = (edge_attr - x_min) / (x_max - x_min)
        x_normalized = edge_attr / x_max
        return x_normalized

    def build_edge_index_and_attr_of_day(self, day: str):
        """
        This function is to build the edge_index and edge_attr.
        Return edge_index and edge_attr as two arrays
        """
        if self._num_edge_features == 1:
            if day not in self.edge_of_each_day or len(self.edge_of_each_day[day])==0:    # there is no edges on this day
                edge_idx = np.ones([2,self.num_nodes], dtype=int)
                edge_idx[:,range(self.num_nodes)] = range(self.num_nodes)   # edge_index only have self loops, i.e. [[0,0],[1,1],[2,2]...]
                edge_idx_T = edge_idx.T
                edge_attr = np.ones([self.num_nodes, 1], dtype=int)   # edge_attr are ones
                # edge_idx_T = None
                # edge_attr = None
            else:
                if self._dynamic == "cooccur":      # 只有在单独使用cooccur 建图时才用过去5交易日内所有的信息
                    # 这里function的day输入是交易日，我们找到包括本交易日在内的前5个交易日（如果其中有自然日则也包括进来），根据这些天内的edge_of_day建立节点之间的连接
                    prev_natual_days = self.get_prev_days(day, prev_len=20, type="natual")
                    # 将prev_natual_days中每天的edge_of_day 放在一个list中
                    prev_edge_of_days = []
                    for i in range(len(prev_natual_days)):
                        prev_edge_of_days.append(self.edge_of_each_day[prev_natual_days[i]])
                    # 有权重地加和不同日期中出现过的相同的edge
                    edge_of_day = self.weighted_sum_dicts(prev_edge_of_days)
                else:       # 其余的建图方法，例如corr, corrNcooccur等，均只用当前交易日的信息
                    edge_of_day = self.edge_of_each_day[day]       # dict saving edges of this day. Key is the pair of firms, value is the count of co-occurance on today. E.g. {('aapl', 'goog'): 15}
                edge_idx_T = None
                edge_attr = None
                for (s_low, t_low) in edge_of_day:
                    s_up = s_low.upper()
                    t_up = t_low.upper()
                    if s_up not in self.valid_firm_list or t_up not in self.valid_firm_list:      # check whether there are firms not in our list
                        raise ValueError(f"One of the firms {(s_up, t_up)} not in valid_firm_list.")
                    else:
                        if edge_of_day[(s_low, t_low)] < self._threshold:  
                            pass    # do not build an edge if the co-occurance is less than the threshold today
                        else:
                            s_idx = self.ticker_to_index(s_up)              # convert source ticker to its index
                            t_idx = self.ticker_to_index(t_up)              # convert target ticker to its index
                            curr_edge = np.array([[s_idx, t_idx]])    # current edge, 
                            if edge_idx_T is None:
                                edge_idx_T = curr_edge                 
                            else:
                                edge_idx_T = np.concatenate([edge_idx_T, curr_edge])
                            if self._weighted_edge:
                                curr_attr = np.array([[edge_of_day[(s_low, t_low)]]])
                            else:
                                curr_attr = np.array([[1]])
                            if edge_attr is None:
                                edge_attr = curr_attr
                            else:
                                edge_attr = np.concatenate([edge_attr, curr_attr])
        else:
            raise ValueError(f"Wrong value: nu_edge_feature = {self._num_edge_features}")
        # if edge_attr is not None:
        #     edge_attr = self.normalize_attr(edge_attr)
        return edge_idx_T, edge_attr

    
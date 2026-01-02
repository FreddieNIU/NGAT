# NGAT

The official repository for paper ["NGAT: A Node-level Graph Attention Network for Long-term Stock Prediction"](https://link.springer.com/chapter/10.1007/978-3-032-04555-3_17) at ICANN 2025

## Data

The ACL2018 dataset can be obtained from: [here](https://github.com/yumoxu/stocknet-dataset)

The SPNews dataset can be obtained from: [here](https://github.com/FreddieNIU/Financial-Graph-Evaluation/tree/main)

After downloading, place it under the `./data` directory with the following structure:

```text
NGAT/
└── code/
└── log/
└── saved/
└── data/
    └── raw/
        └── ACL2018/
            └── price/
            └── tweet/
            └── StockTable/
        └── SPNews/
            └── price/
            └── news/
            └── ReturnMatrix.csv/
            └── company_list.npy/
            └── sp500index.csv/
            


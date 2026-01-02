import sys 
sys.path.append("../code/graph_building/") 
sys.path.append("../code/models/")
import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from firmgraph_dataset import FirmRelationGraph
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

parser.add_argument('--epoch', type=int, default='300',
                    help='Number of epochs')
parser.add_argument('--lr', type=float, default='0.0001',
                    help='Learning Rate')
parser.add_argument('--save',action='store_true',
                    help='default is False, set True if want to save the model')
parser.add_argument('--load_path',type=str, default='No',
                    help='path of the model file to be loaded')
parser.add_argument('--inference',action='store_true',
                    help='default is False, set True if want to skip training and make inference only')

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
    "name" : "SPNews"
}

SAVE_FP = str(os.path.join(       
            ROOT_FOLDER+"saved", 
            graph_params["name"],
            "dynamic-"+str(graph_params["dynamic"]), 
            "threshold-"+str(graph_params["threshold"]),
            "weighted_edge-"+str(graph_params["weighted_edge"]),
            "edge_feature-"+str(graph_params["num_edge_features"]), 
            "Y_feature-"+str(graph_params["num_Y_features"])+"_Ytype-"+graph_params["Y_type"],
            "node_feature-"+str(graph_params["num_node_features"])+"_norm-"+graph_params["node_normalize"]+"_matrix-"+str(graph_params["matrix_node_features"])+"_ESwindow-"+str(graph_params["estimation_window"])))

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


        # model = LSTM(
        #     in_feature=graph_params['num_node_features'],
        #     out_feature=graph_params['num_Y_features'], 
        #     batch_size=len(dataset.graphgenerator.valid_firm_list)
        # )

        # model = AD_GAT(num_stock=dataset[0].x.shape[0], d_market = graph_params['num_node_features'],d_news= 1,
        #               d_hidden = graph_params['num_node_features'], hidn_rnn = 360, heads_att = 6,
        #               hidn_att= 60, dropout = 0.6,t_mix = 0,
        #               infer = 1, relation_static = 1
        #                )
        
        # model = LSTM_GCN(
        #     in_feature=graph_params['num_node_features'],
        #     out_feature=graph_params['num_Y_features'], 
        #     num_nodes=dataset[0].x.shape[0],
        #     edge_dim=graph_params['num_edge_features'], 
        #     matrix_in=graph_params['matrix_node_features'], 
        #     batch_size=len(dataset.graphgenerator.valid_firm_list)
        # )

        # model = TGC(
        #     in_feature=graph_params['num_node_features'],
        #     out_feature=graph_params['num_Y_features'], 
        #     num_nodes=dataset[0].x.shape[0],
        #     edge_dim=graph_params['num_edge_features'], 
        #     matrix_in=graph_params['matrix_node_features'], 
        #     batch_size=len(dataset.graphgenerator.valid_firm_list),
        #     explicit=False,
        #     device=device
        # )

        # model = THGNN(
        #     in_features=graph_params['num_node_features'],
        #     out_features=graph_params['num_Y_features'],
        #     device=device
        # )

        model.to(device)
    else:
        model = torch.load(args.load_path)
        model.to(device)
    
    ## Logger 
    timestamp = time.strftime("-%Y%m%d-%H%M%S", time.localtime())  #time stamp
    SAVED_MODEL = SAVE_FP+f"/Epoch-{args.epoch}_lr-{args.lr}_GAT_{str(timestamp)}.pt"
    logger = Logger(target=graph_params['Y_type'], filename=timestamp)
    logger.write(f"Epoch = {args.epoch}, lr = {args.lr}, save = {args.save}, load_path = {args.load_path} \n")
    
    ## Train, Dev, Test split        
    ## For ACL2018 --------------------------------------------------------
    dev_len, test_len = int(len(dataset)/10), int(len(dataset)/10)
    train_len = len(dataset)-dev_len-test_len
    ## ## -----------------------------------------------------------------
    
    # flash_crash = 9     ## On 2015-08-24, there is a biggest "flash crash" in America stock market, so we remove the period affected by it out of our validation set.
    # dev_len, test_len = int(len(dataset)/10*1.5), int(len(dataset)/10*1.5)-flash_crash
    # train_len = len(dataset)-dev_len-test_len-flash_crash

    if args.inference is False:     # Do training
        ## Optimizer, Learning Rate Deay,  Loss Function, SummaryWriter, Early Stop
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        lr_scheduler = LRScheduler(optimizer, patience=20, min_lr=5e-6, factor=0.5)
        criterion = nn.BCELoss() if graph_params["classification"] else nn.MSELoss() 
        criterion.to(device)
        temp_str = "classification" if graph_params['classification'] else "regression"
        writer = SummaryWriter(log_dir=ROOT_FOLDER+"tensorboard/"+str(temp_str)+"/"+str(graph_params['Y_type'])+"/"+str(timestamp)+'/')
        early_stopping = EarlyStopping(patience=30, min_delta=0)

        ## Training and validation
        best_val_loss = None
        for epoch in tqdm(range(args.epoch)):
            ## Train
            model.train()
            train_loss = 0
            for i in range(train_len):
                data = dataset[i].to(device)
                model.train()
                optimizer.zero_grad()
                out, atten_weight2 = model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            if epoch%20 == 0:
                logger.write(f'epoch: {epoch} \n')
                logger.write(f'Train Loss: {train_loss/train_len} \n')
            writer.add_scalar(tag='Loss/train', scalar_value=train_loss/train_len, global_step=epoch)
            writer.add_scalar(tag='Attention/train', scalar_value=atten_weight2[0][0], global_step=epoch)
            ## Validation
            model.eval()
            dev_loss = 0
            train_evalator = Evaluaor(classification=graph_params['classification'], num_Y_features=graph_params['num_Y_features'], num_data_points=train_len)
            dev_evaluator = Evaluaor(classification=graph_params['classification'], num_Y_features=graph_params['num_Y_features'], num_data_points=dev_len)
            for i in range(train_len):
                data = dataset[i].to(device)
                out, _ = model(data)
                train_evalator.evaluate(pred=out, true=data.y)
            train_pred_true = torch.concat([out,data.y], dim=1)
            for i in range(dev_len):
                data = dataset[train_len+i].to(device)
                out, atten_weight2 = model(data)
                loss = criterion(out, data.y)
                dev_loss += loss.item()
                dev_evaluator.evaluate(pred=out, true=data.y)
            dev_pred_true = torch.concat([out,data.y], dim=1)
            if epoch%20 == 0:
                logger.write(f'Dev Loss: {dev_loss/dev_len} \n')
            ## Apply learing rate decay and early stopping ---------------------------------
            lr_scheduler(dev_loss/dev_len)
            early_stopping(dev_loss/dev_len)
            if early_stopping.early_stop:
                break
            ## --------------------------------------------------------------------------------
            
            ## Save model with the lowest loss --------------------------------------------
            # if epoch%20 == 0:
            #     if best_val_loss is None:
            #         best_val_loss = dev_loss/dev_len
            #         ## Save Model
            #         if args.save:
            #             os.makedirs(SAVE_FP, exist_ok=True)
            #             torch.save(model, SAVED_MODEL)
            #             logger.write(f"Model saved at: {SAVED_MODEL} \n")
            #     elif dev_loss/dev_len < best_val_loss:
            #         best_val_loss = dev_loss/dev_len
            #         ## Save Model
            #         if args.save:
            #             os.makedirs(SAVE_FP, exist_ok=True)
            #             torch.save(model, SAVED_MODEL)
            #             logger.write(f"Model saved at: {SAVED_MODEL} \n")
             ## --------------------------------------------------------------------------------
            writer.add_scalar(tag='Loss/dev', scalar_value=dev_loss/dev_len, global_step=epoch)
            writer.add_scalar(tag='Attention/dev', scalar_value=atten_weight2[0][0], global_step=epoch)
            train_r2,train_mse = train_evalator.average()
            dev_r2,dev_mse = dev_evaluator.average()
            add_scalar_for_multiple_targets_regression(writer=writer, tag='train', r2=train_r2, mse=train_mse, num_Y_feature=graph_params['num_Y_features'], Y_type=graph_params['Y_type'], global_step=epoch)
            add_scalar_for_multiple_targets_regression(writer=writer, tag='dev', r2=dev_r2, mse=dev_mse, num_Y_feature=graph_params['num_Y_features'], Y_type=graph_params['Y_type'], global_step=epoch)
    
        ## Save Model
        if args.save:
            # if dev_loss/dev_len < best_val_loss:
            os.makedirs(SAVE_FP, exist_ok=True)
            torch.save(model, SAVED_MODEL)
            logger.write(f"Model saved at: {SAVED_MODEL} \n")

        ## Load the best model for test
        if args.save:
            # if dev_loss/dev_len < best_val_loss:
            #     pass
            # else:
            logger.write(f"Model loadeed from {SAVED_MODEL}")
            model = torch.load(SAVED_MODEL)

    else:   ## Inference only
        best_val_loss = None
        if args.load_path == "No":
            try:
                model = torch.load(SAVED_MODEL)
                model.to(device)
                logger.write(f'Inference Only! Model loaded from {SAVED_MODEL} \n')
            except FileNotFoundError:
                raise FileNotFoundError("Doing inference without model load path provided! Please give load_path")
        else:
            logger.write(f'Inference Only! Model loaded from {args.load_path} \n')
        ## Inference on Train and Dev sets
        model.eval()
        dev_loss = 0
        train_evalator = Evaluaor(classification=graph_params['classification'],num_Y_features=graph_params['num_Y_features'], num_data_points=train_len)
        dev_evaluator = Evaluaor(classification=graph_params['classification'], num_Y_features=graph_params['num_Y_features'], num_data_points=dev_len)
        for i in range(train_len):
            data = dataset[i].to(device)
            out, _ = model(data)
            train_evalator.evaluate(pred=out, true=data.y)
        train_pred_true = torch.concat([out,data.y], dim=1)
        for i in range(dev_len):
            data = dataset[train_len+i].to(device)
            out, _ = model(data)
            dev_evaluator.evaluate(pred=out, true=data.y)
        dev_pred_true = torch.concat([out,data.y], dim=1)
        train_r2,train_mse = train_evalator.average()
        dev_r2,dev_mse = dev_evaluator.average()

    
    ## Evaluate on Test
    model.eval()
    test_evaluator = Evaluaor(classification=graph_params['classification'], num_Y_features=graph_params['num_Y_features'], num_data_points=test_len)
    atten_weight_test = []
    for i in tqdm(range(test_len)):
        data = dataset[train_len+dev_len+i]
        data.to(device)
        out, atten_weight = model(data)
        atten_weight_test.append(rescale_tensor(atten_weight))
        test_evaluator.evaluate(pred=out, true=data.y)
    test_pred_true = torch.concat([out,data.y], dim=1)
    logger.write(f"Train Pred & True: {train_pred_true} \n")
    logger.write(f"Dev Pred & True: {dev_pred_true} \n")
    logger.write(f"Test Pred & True: {test_pred_true} \n")
    test_r2,test_mse = test_evaluator.average()
    logger.write(f"Train R2: {train_r2}, Train MSE: {train_mse}  \n")
    logger.write(f"Val R2: {dev_r2}, Val MSE: {dev_mse}  \n")
    logger.write(f"Test R2: {test_r2}, Test MSE: {test_mse}  \n")
    
    if args.inference:
        logger.write(f"Test attention 0: {atten_weight_test[0]} \n")
        # logger.write(f"Test attention 1: {atten_weight_test[1]} \n")
        # logger.write(f"Test attention 2: {atten_weight_test[2]} \n")
        plot_heatmap(matrix=atten_weight_test[0], save_path="Reg_ACL_NGAT_attn_0_new.png")
        # plot_heatmap(matrix=atten_weight_test[1], save_path="Reg_ACL_GAT_attn_1.png")
        # plot_heatmap(matrix=atten_weight_test[2], save_path="Reg_ACL_GAT_attn_2.png")


    logger.write(f"Graph Parameters: {graph_params} \n")
    logger.write(f"Model: {model} \n")
    try:
        logger.write(f"Hidden: {model.hidden}, Head: {model.out_head}, drouput rate: {model.dropout}")
    except AttributeError:
        pass
    

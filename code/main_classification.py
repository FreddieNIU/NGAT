import sys 
sys.path.append("../code/graph_building/") 
sys.path.append("../code/models/")
import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from firmgraph_dataset import FirmRelationGraph
from GAT import GAT
from GAT_classification import GAT_C
from ADGAT import AD_GAT
from LSTM import LSTM
from LSTM_GCN import LSTM_GCN
from TGC import TGC
from evaluator import Evaluaor
from torch.utils.tensorboard import SummaryWriter
from utils import *

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
parser.add_argument('--save_output',action='store_true',
                    help='default is False, set True if want to save the model output')

ROOT_FOLDER = "../"

graph_params = {
    ##  -------- Graph Params ---------
    "dynamic" : "cooccur",
    ##  -------- Node Params ---------
    "num_node_features" : 8,
    "matrix_node_features" : True,
    "node_normalize" : 'mean',
    "num_Y_features" : 4,
    "Y_type" : "mean",
    "classification" : True,
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
    # device = torch.device("cpu")
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
        model_name = "ngat"
        model = GAT_C(
            in_feature=graph_params['num_node_features'],
            out_feature=graph_params['num_Y_features'], 
            num_nodes=dataset[0].x.shape[0],
            edge_dim=graph_params['num_edge_features'], 
            matrix_in=graph_params['matrix_node_features'], 
            batch_size=len(dataset.graphgenerator.valid_firm_list),
            device=device
        )
        
        # model_name = "lstm"
        # model = LSTM(
        #     in_feature=graph_params['num_node_features'],
        #     out_feature=graph_params['num_Y_features'], 
        #     batch_size=len(dataset.graphgenerator.valid_firm_list)
        # )
        # model_name = "adgat"
        # model = AD_GAT(num_stock=dataset[0].x.shape[0], d_market = graph_params['num_node_features'],d_news= 1,
        #               d_hidden = graph_params['num_node_features'], hidn_rnn = 360, heads_att = 6,
        #               hidn_att= 60, dropout = 0.6,t_mix = 1,
        #               infer = 1, relation_static = 0
        #                )

        # model_name = "tgc"
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
        
        # model = LSTM_GCN(
        #     in_feature=graph_params['num_node_features'],
        #     out_feature=graph_params['num_Y_features'], 
        #     num_nodes=dataset[0].x.shape[0],
        #     edge_dim=graph_params['num_edge_features'], 
        #     matrix_in=graph_params['matrix_node_features'], 
        #     batch_size=len(dataset.graphgenerator.valid_firm_list)
        # )

        model.to(device)
    else:
        model = torch.load(args.load_path)
        model.to(device)
    
    ## Logger 
    timestamp = time.strftime("-%Y%m%d-%H%M%S", time.localtime())  #time stamp
    SAVED_MODEL = SAVE_FP+f"/Epoch-{args.epoch}_lr-{args.lr}_GAT_{str(timestamp)}.pt"
    logger = Logger(target=graph_params['Y_type'], filename=timestamp)
    logger.write(f"Graph Parameters: {graph_params} \n")
    logger.write(f"Epoch = {args.epoch}, lr = {args.lr}, save = {args.save}, load_path = {args.load_path} \n")
    
    ## Train, Dev, Test split
    dev_len, test_len = int(len(dataset)/10), int(len(dataset)/10)
    train_len = len(dataset)-dev_len-test_len

    # flash_crash = 9     ## On 2015-08-24, there is a biggest "flash crash" in America stock market, so we remove the period affected by it out of our validation set.
    # dev_len, test_len = int(len(dataset)/10*1.5), int(len(dataset)/10*1.5)-flash_crash
    # train_len = len(dataset)-dev_len-test_len-flash_crash

    if args.inference is False:     # Do training
        ## Optimizer, Learning Rate Deay,  Loss Function, SummaryWriter, Early Stop
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        lr_scheduler = LRScheduler(optimizer, patience=20, min_lr=1e-6, factor=0.5)
        criterion = nn.BCELoss() if graph_params["classification"] else nn.MSELoss() 
        criterion.to(device)
        temp_str = "classification" if graph_params['classification'] else "regression"
        writer = SummaryWriter(log_dir=ROOT_FOLDER+"tensorboard/"+str(temp_str)+"/"+str(graph_params['Y_type'])+"/"+str(timestamp)+'/')
        early_stopping = EarlyStopping(patience=30, min_delta=0)

        ## Training and validation
        for epoch in tqdm(range(args.epoch)):
            ## Train
            model.train()
            train_loss = 0
            for i in range(train_len):
                data = dataset[i].to(device)
                model.train()
                optimizer.zero_grad()
                out, atten_weight2 = model(data)
                # y_onehot = torch.nn.functional.one_hot(data.y.long().squeeze(), num_classes=2)
                # print(out.shape,  y_onehot.shape)
                # print(out[:5],  y_onehot[:5])
                loss = criterion(out, data.y.double())  #这里将linear层输出的结果logit直接输入到cross entropy loss而不是经过了softmax层的out，因为loss自带softmax操作
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            if epoch%20 == 0:
                logger.write(f'epoch: {epoch} \n')
                logger.write(f'Train Loss: {train_loss/train_len} \n')
            writer.add_scalar(tag='Loss/train', scalar_value=train_loss/train_len, global_step=epoch)
            # writer.add_scalar(tag='Attention/train', scalar_value=atten_weight2[0][0], global_step=epoch)
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
                out,  atten_weight2 = model(data)
                # y_onehot = torch.nn.functional.one_hot(data.y.long().squeeze(), num_classes=2)
                loss = criterion(out, data.y.double())
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
            writer.add_scalar(tag='Loss/dev', scalar_value=dev_loss/dev_len, global_step=epoch)
            # writer.add_scalar(tag='Attention/dev', scalar_value=atten_weight2[0][0], global_step=epoch)
            train_acc,train_mcc,train_auc = train_evalator.average()
            dev_acc,dev_mcc,dev_auc = dev_evaluator.average()
            add_scalar_for_multiple_targets_classification(writer=writer, tag='train', acc=train_acc, mcc=train_mcc, auc=train_auc, num_Y_feature=graph_params['num_Y_features'], Y_type=graph_params['Y_type'], global_step=epoch)
            add_scalar_for_multiple_targets_classification(writer=writer, tag='dev', acc=dev_acc, mcc=dev_mcc, auc=dev_auc, num_Y_feature=graph_params['num_Y_features'], Y_type=graph_params['Y_type'], global_step=epoch)
    else:   ## Inference only
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
        train_evalator = Evaluaor(classification=graph_params['classification'], num_Y_features=graph_params['num_Y_features'], num_data_points=train_len)
        dev_evaluator = Evaluaor(classification=graph_params['classification'], num_Y_features=graph_params['num_Y_features'], num_data_points=dev_len)
        for i in range(train_len):
            data = dataset[i].to(device)
            out,_ = model(data)
            train_evalator.evaluate(pred=out, true=data.y)
        train_pred_true = torch.concat([out,data.y], dim=1)
        for i in range(dev_len):
            data = dataset[train_len+i].to(device)
            out,_= model(data)
            dev_evaluator.evaluate(pred=out, true=data.y)
        dev_pred_true = torch.concat([out,data.y], dim=1)
        train_acc,train_mcc,train_auc = train_evalator.average()
        dev_acc,dev_mcc,dev_auc = dev_evaluator.average()

    ## Save Model
    if args.save:
        os.makedirs(SAVE_FP, exist_ok=True)
        torch.save(model, SAVED_MODEL)
        logger.write(f"Model saved at: {SAVED_MODEL} \n")

    ## Evaluate on Test
    model.eval()
    if args.save_output:
        predSaver = OutputSaver(name=graph_params['name']+"_pred", model=model_name)
        trueSaver = OutputSaver(name=graph_params['name']+"_true", model=model_name)
        lastPeriodSaver = OutputSaver(name=graph_params['name']+"_last", model=model_name)
        last_period = dataset.graphgenerator.last_period
        keys = list(last_period.keys())
    test_evaluator = Evaluaor(classification=graph_params['classification'], num_Y_features=graph_params['num_Y_features'], num_data_points=test_len)
    atten_weight_test = []
    for i in tqdm(range(test_len)):
        data = dataset[train_len+dev_len+i]
        data.to(device)
        out,atten_weight = model(data)
        atten_weight_test.append(atten_weight)
        if args.save_output:
            predSaver.record_step_output(out)
            trueSaver.record_step_output(data.y)
            lastPeriodSaver.record_step_output(torch.tensor(last_period[keys[i]]))
        test_evaluator.evaluate(pred=out, true=data.y)
    test_pred_true = torch.concat([out,data.y], dim=1)
    if args.save_output:
        predSaver.cat_and_save_final_output()
        trueSaver.cat_and_save_final_output()
        lastPeriodSaver.cat_and_save_final_output()
    logger.write(f"Train Pred & True: {train_pred_true} \n")
    logger.write(f"Dev Pred & True: {dev_pred_true} \n")
    logger.write(f"Test Pred & True: {test_pred_true} \n")
    test_acc,test_mcc, test_auc = test_evaluator.average()
    logger.write(f"Train Accuracy: {train_acc}, Train MCC: {train_mcc}, Train AUC: {train_auc}  \n")
    logger.write(f"Val Accuracy: {dev_acc}, Val MCC: {dev_mcc}, Val AUC: {dev_auc}  \n")
    logger.write(f"Test Accuracy: {test_acc}, Test MCC: {test_mcc}, Test AUC: {test_auc}  \n")

    logger.write(f"Graph Parameters: {graph_params} \n")
    logger.write(f"Model: {model} \n")
    try:
        logger.write(f"Hidden: {model.hidden}, Head: {model.out_head}, drouput rate: {model.dropout}")
    except AttributeError:
        pass

    if args.inference:
        logger.write(f"Test attention 0: {atten_weight_test[0]} \n")
        logger.write(f"Test attention 1: {atten_weight_test[1]} \n")
        logger.write(f"Test attention 2: {atten_weight_test[2]} \n")
        plot_heatmap(matrix=atten_weight_test[0], save_path="Cla_SPN_NGAT_attn_0.png")
        plot_heatmap(matrix=atten_weight_test[1], save_path="Cla_SPN_NGAT_attn_1.png")
        plot_heatmap(matrix=atten_weight_test[2], save_path="Cla_SPN_NGAT_attn_2.png")

    

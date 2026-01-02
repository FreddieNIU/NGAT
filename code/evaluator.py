from torcheval.metrics.functional import mean_squared_error, r2_score, binary_accuracy, binary_auroc, binary_f1_score, binary_confusion_matrix
import torch
cpu = torch.device('cpu')
class Evaluaor():

    def __init__(self, classification:bool, num_Y_features:int, num_data_points:int) -> None:
        self.classification = classification
        self.num_Y_features = num_Y_features
        self.num_data_points = num_data_points
        if self.classification:
            self.acc, self.mcc, self.auc = self.init_eval_metrix()
        else:   
            self.r2, self.mse = self.init_eval_metrix()
        
        self.pred_saver = []    ## this is only for graph-level tasks. For node-level tasks, this variable is useless.
        self.true_saver = []    ## this is only for graph-level tasks. For node-level tasks, this variable is useless.


    def init_eval_metrix(self):
        """
        Create the initial variable to save evaluation metrix. R2 and MSE are used.
        """
        if self.classification:
            if self.num_Y_features == 1:
                acc, mcc, auc = [], [], []
            else:
                acc, mcc, auc  = [], [], []

            return acc, mcc, auc
        else:
            if self.num_Y_features == 1:
                r2, mse = 0, 0
            else:
                r2, mse = [], []

            return r2, mse

    def Matthews_Correlation_Coefficient(self, pred, true):
        confusion_matrix = binary_confusion_matrix(pred, true)
        tp, fn, fp, tn = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]
        numerator =  (tp * tn - fp * fn)
        denominator = torch.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        if denominator == 0:
            mcc = 0.0
        else:
            mcc = numerator / denominator
        return mcc

    def evaluate(self, pred, true):
        """
        Given the number of Y features
        pred: predicted output
        true: true target value
        """
        if self.classification:
            true = true.int()
            if self.num_Y_features == 1:
                # pred = pred[:,0].le(0.5).long()
                pred = pred[:,0]
                acc = binary_accuracy(pred, true[:,0])
                mcc = self.Matthews_Correlation_Coefficient(pred=pred, true=true[:,0])
                auc = binary_auroc(pred, true[:,0])
                self.acc.append(acc)
                self.mcc.append(mcc)
                self.auc.append(auc)
            else:
                acc, mcc, auc = [], [], []
                for col in range(self.num_Y_features):
                    acc.append(binary_accuracy(pred[:,col], true[:,col]))
                    mcc.append(self.Matthews_Correlation_Coefficient(pred=pred[:,col], true=true[:,col]))
                    auc.append(binary_auroc(pred[:,col], true[:,col]))
                self.acc.append(acc)
                self.mcc.append(mcc)
                self.auc.append(auc)
        else:
            if self.num_Y_features == 1:
                r2 = r2_score(pred, true)
                mse = mean_squared_error(pred, true)
                self.r2 += r2
                self.mse += mse
            else:
                r2, mse = [], []
                for col in range(self.num_Y_features):
                    r2.append(r2_score(pred[:,col], true[:,col]))
                    mse.append(mean_squared_error(pred[:,col], true[:,col]))
                self.r2.append(r2)
                self.mse.append(mse)

        
    def average(self):
        """
        Average the metrix along the same target
        If num_Y_features != 1, 
        E.g. 3, then we return r2_average = [r2_of_rmean5, r2_of_rmean10, r2_of_rmean20], similar for mse_average
        """
        if self.classification:
            if self.num_Y_features == 1:
                acc_average = torch.mean(torch.tensor(self.acc), dim=0)
                mcc_average = torch.mean(torch.tensor(self.mcc), dim=0)
                auc_average = torch.mean(torch.tensor(self.auc), dim=0)
            else:
                acc_average = torch.mean(torch.tensor(self.acc), dim=0)
                mcc_average = torch.mean(torch.tensor(self.mcc), dim=0)
                auc_average = torch.mean(torch.tensor(self.auc), dim=0)
            self.acc_average = acc_average
            self.mcc_average = mcc_average
            self.auc_average = auc_average

            self.acc, self.mcc, self.auc = self.init_eval_metrix()
            return self.acc_average, self.mcc_average, self.auc_average
        else:
            if self.num_Y_features  == 1:
                r2_average = self.r2 / self.num_data_points
                mse_average = self.mse / self.num_data_points
            else:
                r2_average = torch.mean(torch.tensor(self.r2), dim=0)
                mse_average = torch.mean(torch.tensor(self.mse), dim=0)
        
            self.r2_average = r2_average
            self.mse_average = mse_average
            
            self.r2, self.mse = self.init_eval_metrix()
            return self.r2_average, self.mse_average
        
    
    def record(self, pred, true):
        """
        pred: predicted output, shape = [num_y_features]
        true: true target value, shape = [num_y_features]
        """
        pred = pred.detach().cpu()
        true = true.detach().cpu()

        self.pred_saver.append(pred)
        self.true_saver.append(true)

    def stack(self):
        """
        stack the pred_saver and true_saver from list to tensor. 
        """
        self.pred_saver = torch.stack(self.pred_saver)
        self.true_saver = torch.stack(self.true_saver)
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


ROOT_FOLDER = "../"

class Logger(object):
    def __init__(self, target:str, filename:str):
        self.log_folder = ROOT_FOLDER+"log/" + target
        os.makedirs(self.log_folder, exist_ok=True)
        self.log = open(self.log_folder+"/"+filename+".log", 'a')


    def write(self, message):
        self.log.write(message)

    def flush(self):
        pass


def add_scalar_for_multiple_targets_regression(writer, num_Y_feature:int, Y_type:str, tag, r2, mse, global_step):
    """
    This function is to add_scalars in the TensorBoard since we may have more than 1 target in Y.
    """
    valid_tags = ['train', 'dev', 'test']
    if num_Y_feature == 1:
        if tag in valid_tags:
            writer.add_scalar(tag=f"R2/{tag}", scalar_value=r2, global_step=global_step)
            writer.add_scalar(tag=f"MSE/{tag}", scalar_value=mse, global_step=global_step)
        else:
            raise ValueError(f"Wrong value for metrix {tag}")
    else:
        if Y_type == 'mean':
            targets = ['rmean5', 'rmean10','rmean20']
        elif Y_type == 'std':
            targets = ['std5', 'std10','std20']
        else:
            targets = ['rmean5', 'std5', 'rmean10','std10','rmean20','std20']
        if tag in valid_tags:
            for t in range(len(targets)):
                writer.add_scalar(tag=f"R2/{tag}_{targets[t]}", scalar_value=r2[t], global_step=global_step)
                writer.add_scalar(tag=f"MSE/{tag}_{targets[t]}", scalar_value=mse[t], global_step=global_step)
        else:
            raise ValueError(f"Wrong value for metrix {tag}")

def add_scalar_for_multiple_targets_classification(writer, num_Y_feature:int, Y_type:str, tag, acc, mcc, auc, global_step):
    """
    This function is to add_scalars in the TensorBoard since we may have more than 1 target in Y.
    """
    valid_tags = ['train', 'dev', 'test']
    if num_Y_feature == 1:
        if tag in valid_tags:
            writer.add_scalar(tag=f"ACC/{tag}", scalar_value=acc, global_step=global_step)
            writer.add_scalar(tag=f"F1/{tag}", scalar_value=mcc, global_step=global_step)
            writer.add_scalar(tag=f"AUC/{tag}", scalar_value=auc, global_step=global_step)
        else:
            raise ValueError(f"Wrong value for metrix {tag}")
    else:
        if Y_type == 'mean':
            targets = ['rmean5', 'rmean10','rmean20']
        elif Y_type == 'std':
            targets = ['std5', 'std10','std20']
        else:
            targets = ['rmean1', 'rmean5', 'rmean10','rmean20']
        if tag in valid_tags:
            for t in range(len(targets)):
                writer.add_scalar(tag=f"ACC/{tag}_{targets[t]}", scalar_value=acc[t], global_step=global_step)
                writer.add_scalar(tag=f"MCC/{tag}_{targets[t]}", scalar_value=mcc[t], global_step=global_step)
                writer.add_scalar(tag=f"AUC/{tag}_{targets[t]}", scalar_value=auc[t], global_step=global_step)
        else:
            raise ValueError(f"Wrong value for metrix {tag}")

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # self.best_loss = val_loss

            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

    
class OutputSaver:
    def __init__(self, name='pred', model="lstm"):
        # List to store the outputs from each step
        self.outputs = []
        self.name = name
        self.model = model

    def record_step_output(self, step_output):
        # Record the tensor from a single step
        self.outputs.append(step_output)

    def cat_and_save_final_output(self, columns=None):
        filename = self.name+'.csv'
        folderpath = ROOT_FOLDER+"code/notebook/"+self.model+"/"
        os.makedirs(folderpath, exist_ok=True)

        # Concatenate the recorded outputs along the first dimension
        final_output = torch.cat(self.outputs, dim=0)
        
        # Convert the tensor to a NumPy array
        final_output_np = final_output.cpu().detach().numpy()
        
        # Define column names if not provided
        if columns is None:
            columns = ['rmean1_'+self.name, 'rmean5_'+self.name, 'rmean10_'+self.name, 'rmean21_'+self.name]
        
        # Create a DataFrame and save as a CSV file
        df = pd.DataFrame(final_output_np, columns=columns)
        df.to_csv(folderpath+filename, index=False)

        print(f"Saved concatenated output to '{filename}'")

def plot_heatmap(matrix, save_path=None):
    """
    Plots a heatmap of a given [N, N] matrix and optionally saves it to a file.

    Args:
        matrix (torch.Tensor or np.ndarray): A 2D numpy array or PyTorch tensor of shape [N, N].
        save_path (str, optional): Path to save the heatmap figure. If None, the figure is not saved.
    """

    rcParams.update({'font.size': 16})
    # Check if the input is a PyTorch tensor
    if torch.is_tensor(matrix):
        # Move tensor to CPU if it is on CUDA
        if matrix.is_cuda:
            matrix = matrix.cpu()
        # Convert to numpy array
        matrix = matrix.detach().numpy()
    
    # Ensure the input is a 2D numpy array
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input must be a 2D numpy array or PyTorch tensor with shape [N, N].")

    # Create the heatmap
    plt.imshow(matrix, cmap='Reds', interpolation='nearest')
    plt.colorbar()  # Adds a colorbar to the side of the heatmap

    # Add labels and title
    plt.title('Heatmap of the Attention')
    plt.xlabel('Target Company Index')
    plt.ylabel('Source Company Index')

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(ROOT_FOLDER+"code/notebook/attention_heatmaps/"+save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    # Show the plot
    plt.show()


def rescale_tensor(tensor):
    """
    Rescales a 2D tensor to the range [0, 1].

    Args:
        tensor (torch.Tensor): A 2D tensor of shape [N, N].

    Returns:
        torch.Tensor: The rescaled tensor with values in the range [0, 1].
    """
    # Ensure the tensor is of type float for proper scaling
    tensor = tensor.float()

    # Find the minimum and maximum values of the tensor
    min_val = tensor.min()
    max_val = tensor.max()

    # Avoid division by zero (if all values in tensor are the same)
    if max_val > min_val:
        # Rescale to [0, 1]
        rescaled_tensor = (tensor - min_val) / (max_val - min_val)
    else:
        # If max_val equals min_val, return a tensor of zeros (or ones, depending on your preference)
        rescaled_tensor = torch.zeros_like(tensor)

    return rescaled_tensor
import torch
from torch.optim import lr_scheduler

import yaml
import pprint
import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

def print_and_log(string, log, printOut=True):
    if printOut:
        print('{}'.format(string))
    with open(log, 'a') as l:
        l.write('{}\n'.format(string))
        l.flush()


def load_config():
    ''' Load config file from command line '''

    # Get the config path 
    parser = ArgumentParser()
    parser.add_argument('-c', '--config',
                        dest='config_path',
                        help='config file path',
                        required=True)
    
    path = parser.parse_args().config_path
    # Load the config file
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Print the config file contents
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    return config

def to_one_hot(inp, n_classes):
    '''
    Index target to one-hot
    From: https://github.com/vikasverma1077/manifold_mixup/blob/master/supervised/models/utils.py
    '''
    y_onehot = torch.FloatTensor(inp.size(0), n_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    return y_onehot

def get_lambda(alpha=1.0):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

def mixup_process(inp, target_reweighted, lam):
    '''
    Create the batch mixup inputs and reweighted targets
    From: https://github.com/vikasverma1077/manifold_mixup/blob/master/supervised/utils.py
    '''
    indices = np.random.permutation(inp.size(0))
    inp = inp*lam + inp[indices]*(1-lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    return inp, target_reweighted

def save_checkpoint(state, save_path, is_best):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, save_path+'checkpoint.pth.tar')

    # Save model with best val acc
    if is_best:
        torch.save(state, save_path + 'best_model.pth.tar')

def calculate_time(start, end):
    ''' Calculate and return the hours, minutes and seconds between start and end times '''
    hours, remainder = divmod(end-start, 60*60)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), seconds

class AverageMeter(object):
    ''' 
    Computes and stores the average and current value 
    From: https://github.com/vikasverma1077/manifold_mixup/blob/master/supervised/utils.py
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class RecorderMeter(object):
    ''' 
    Computes and stores the minimum loss value and its epoch index 
    From: https://github.com/vikasverma1077/manifold_mixup/blob/master/supervised/utils.py
    '''
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0

        self.total_epoch   = total_epoch+1
        self.current_epoch = 0
        self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)

        self.epoch_losses  [idx, 0] = train_loss
        self.epoch_losses  [idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1
        return self.max_accuracy(False) == val_acc

    def expand(self, total_epoch):
        expansion  = np.zeros((total_epoch + 1 - self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
        self.epoch_losses = np.r_[self.epoch_losses, expansion]
        self.epoch_accuracy = np.r_[self.epoch_accuracy, expansion]

        self.total_epoch = total_epoch+1

    def max_accuracy(self, isTrain):
        if self.current_epoch <= 0: return 0
        if isTrain: 
            return self.epoch_accuracy[:self.current_epoch, 0].max()
        else:       
            return self.epoch_accuracy[:self.current_epoch, 1].max()
      
    def plot_curve(self, save_path):
        title = 'Accuracy/Loss curve of Train/Val'
        dpi = 80  
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(1, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Accuracy/Loss', fontsize=16)
      
        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        
        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
          fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
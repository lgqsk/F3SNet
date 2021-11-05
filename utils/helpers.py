import logging
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from utils.dataloaders import (full_path_loader, full_test_loader, full_show_loader, CDDloader)
from utils.metrics import jaccard_loss, dice_loss
from utils.losses import hybrid_loss,WCELoss
# from models.UNetAP3.UNetAP3_V36 import UNetAP3_V36
from models.F3SNet import F3SNet as net

logging.basicConfig(level=logging.INFO)

def initialize_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values

    Returns
    -------
    dict
        a dictionary of metrics

    """
#     metrics = {
#         'cd_losses': [],
#         'cd_TP': [],
#         'cd_FN': [],
#         'cd_FP': [],
#         'cd_TN': [],
#         'learning_rate': [],
#     }
#     return metrics
    return np.array([[0,0],[0,0]])

def get_index(conmetrix):
    [TP,FN],[FP,TN] = conmetrix
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1_score = 2*(Precision*Recall)/(Precision+Recall)
    OA = (TP+TN)/(TP+TN+FP+FN)
    return [F1_score,Precision,Recall,OA]

def get_mean_metrics(metric_dict):
    """takes a dictionary of lists for metrics and returns dict of mean values

    Parameters
    ----------
    metric_dict : dict
        A dictionary of metrics

    Returns
    -------
    dict
        dict of floats that reflect mean metric value

    """
    return {k: np.mean(v) for k, v in metric_dict.items()}


def set_metrics(metric_dict, cd_loss, cd_report, lr):
    """Updates metric dict with batch metrics

    Parameters
    ----------
    metric_dict : dict
        dict of metrics
    cd_loss : dict(?)
        loss value
    cd_corrects : dict(?)
        number of correct results (to generate accuracy
    cd_report : list
        precision, recall, f1 values

    Returns
    -------
    dict
        dict of  updated metrics


    """
    metric_dict['cd_losses'].append(cd_loss.item())
    metric_dict['cd_TP'].append(cd_report[0])
    metric_dict['cd_FN'].append(cd_report[1])
    metric_dict['cd_FP'].append(cd_report[2])
    metric_dict['cd_TN'].append(cd_report[3])
    metric_dict['learning_rate'].append(lr)

    return metric_dict

def set_test_metrics(metric_dict, cd_corrects, cd_report):

    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])

    return metric_dict


def get_loaders(opt):


    logging.info('STARTING Dataset Creation')

    train_full_load, val_full_load = full_path_loader(opt.dataset_dir)


    train_dataset = CDDloader(train_full_load, aug=opt.augmentation)
    val_dataset = CDDloader(val_full_load, aug=False)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return train_loader, val_loader

def get_test_loaders(opt):


#     logging.info('STARTING Dataset Creation')

    test_full_load = full_test_loader(opt.dataset_dir)

    test_dataset = CDDloader(test_full_load, aug=False)

#     logging.info('STARTING Dataloading')


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return test_loader

def get_show_loaders(opt):
    show_full_load = full_show_loader(opt.dataset_dir)

    show_dataset = CDDloader(show_full_load, aug=False)

    show_loader = torch.utils.data.DataLoader(show_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return show_loader

def get_show_loaders_2(dataset_dir="../CDD/Real/subset/",batch_size=1):
    show_full_load = full_show_loader(dataset_dir)

    show_dataset = CDDloader(show_full_load, aug=False)

    show_loader = torch.utils.data.DataLoader(show_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    return show_loader

def get_criterion(opt):
    """get the user selected loss function

    Parameters
    ----------
    opt : dict
        Dictionary of options/flags

    Returns
    -------
    method
        loss function

    """
    if opt.loss_function == 'hybrid':
        criterion = hybrid_loss
    elif opt.loss_function == 'wce':
        criterion = WCELoss()
    elif opt.loss_function == 'dice':
        criterion = dice_loss
    elif opt.loss_function == 'bce':
        criterion = nn.CrossEntropyLoss()
    elif opt.loss_function == 'jaccard':
        criterion = jaccard_loss
    return criterion


def load_model(opt, device):
    
    model = net(opt.image_chanels,opt.init_channels,opt.bilinear).to(device)
    return model

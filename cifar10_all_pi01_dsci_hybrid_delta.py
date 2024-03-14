import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from robustbench.data import load_cifar10c, load_cifar10
from utils.cheap_resnet import ResNet, Bottleneck
from collections import Counter
import utils.thresholding_adaptingspecificchannels as thc_specific
import utils.thresholding_combined as thc

import random
import copy 
import os
import ast 
import csv
import math
import time
import glob
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, unique, concat, value_counts
from numpy import argmax, argmin, array, arange
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import models.tent as tent
from models.tent import check_model
import json
import pickle
from copy import deepcopy
from types import SimpleNamespace
from models.delta import *
from models.delta_hybrid import *
import yaml
from test_utils import *
# from import_dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('PyTorch version:', torch.__version__)

# Configs
'''
In this case, we used weights from Resnet-26 trained on CIFAR10, 
and will test it on CIFAR10_corrupted, 
with adapting using the channel-level information.
'''

#Fixed configs
model_arch = 'Resnet-26'
optimizer = 'SGD'
# batch_size = 500
lr = 0.1
epochs = 200
weight_decay = 5e-4
momentum = 0.9
save_dir = '/home/mila/p/paria.mehrbod/scratch/TTA/test_time_adaptation/outputs/'
adaptation = 'norm_threshold'
dist_choice = 'Wasser'
chosen_layer = 28
tta_epochs = 1

switch = False
scoring = 'sum'

#Model config
parser = argparse.ArgumentParser(description='PyTorch Byra Test-time Adaptation using Channel-level statistics')

#Dataset config
parser.add_argument('--corruption', default = 'gaussian_noise', type=str, help='there are 19 types of corruption. default is gaussian_noise')
parser.add_argument('--severity', default = 0, type=int, help='severity, 0 to 5')
parser.add_argument('--batch_size', default = 500, type=int, help='batch size')
parser.add_argument('--exp_setup', required= True ,choices=["DS+CB","DS+CI","IS+CI","IS+CB","class"], help='distribution of data')
parser.add_argument('--n_classes', type=int, help='number of classes, 1 to 10')
parser.add_argument('--rho', required=False, type=float, help='long tail factor')
parser.add_argument('--pi', required=False, type=float, help='dirichlet factor')
# parser.add_argument('--file',required=True, help='json file path')
args = parser.parse_args()

number_of_classes=10
corruption = args.corruption
severity = args.severity


DATA_PATH='/home/mila/p/paria.mehrbod/scratch/TTA/data/CIFAR-10-C/'
file= f"cifar10_online_b{args.batch_size}_sev{args.severity}_cor{args.corruption}"



parameters={}
if args.exp_setup == "class" and args.n_classes is None:
    parser.error("When exp_setup is 'class', --n_classes is required.")
if args.exp_setup == "class":
    file+=f"_{args.exp_setup}_{args.n_classes}"

if args.exp_setup == "DS+CB" and args.rho is None:
    parser.error("When exp_setup is 'DS+CB', --rho is required.")
if args.exp_setup == "DS+CB":
    file+=f"_{args.exp_setup}_{args.rho}"
    parameters["rho"]=args.rho

if args.exp_setup == "IS+CI" and args.pi is None:
    parser.error("When exp_setup is 'IS+CI', --pi is required.")
if args.exp_setup == "IS+CI":
    file+=f"_{args.exp_setup}_{args.pi}"
    parameters["pi"]=args.pi
    
if args.exp_setup == "DS+CI" and args.rho is None:
    parser.error("When exp_setup is 'DS+CI', --rho is required.")

if args.exp_setup == "DS+CI" and args.pi is None:
    parser.error("When exp_setup is 'DS+CI', --pi is required.")
if args.exp_setup == "DS+CI":
    file+=f"_{args.exp_setup}_rho{args.rho}_pi{args.pi}"
    parameters["rho"]=args.rho
    parameters["pi"]=args.pi

file+=".json"

#Model
def ResNet26():
    return ResNet(Bottleneck, [2, 2, 2, 2], num_classes=10)

def batch_norm_stats(model, print_stats=True):
    bn_layers = 0
    bn_stats = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers += 1
            layer_mean = module.running_mean
            layer_var = module.running_var
            layer_stats = [[layer_mean.min().item(), layer_mean.mean().item(), layer_mean.max().item()],
                           [layer_var.min().item(), layer_var.mean().item(), layer_var.max().item()]]
            bn_stats.append(layer_stats)
    if print_stats:
        print('There are %.d batch_norm layers.' % bn_layers)
        print('Stats for those layers: \nrunning_mean and running_var [min, mean, max]')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in bn_stats]))
    return bn_stats

def list_modules(model):
    module_list = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            module_list.append(copy.deepcopy(m))
    return nn.ModuleList(module_list)


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = int(train_labels.max() + 1)
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    client_idcs = [[[] for _ in range(n_classes)] for _ in range(n_clients)]

    for c, (c_idcs, fracs) in enumerate(zip(class_idcs, label_distribution)):
        for i, idcs in enumerate(np.split(c_idcs, (np.cumsum(fracs)[:-1] * len(c_idcs)).astype(int))):
            client_idcs[i][c].append(idcs)

    client_idcs_final = []
    for client in client_idcs:
        random.shuffle(client)
        client_idcs_final.append(np.concatenate(client, axis=1)[0])

    return client_idcs_final


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.labels)
        

def load_corrupt_data(corruption, severity):
    data = np.load(f'{DATA_PATH}{corruption}.npy')
    labels = np.load(f'{DATA_PATH}labels.npy')
    images = data[(severity-1) * 10000: ((severity-1) * 10000) + 10000,...]
    labels = labels[(severity-1) * 10000: ((severity-1) * 10000) + 10000]
    images = images.transpose(0,3,1,2)
    
    return images, labels

def load_normal_data():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform = transform_test)
    labels = np.array([label for _, label in dataset])
    images = np.array([data.numpy() for data, _ in dataset])
    return images, labels
    
    
        
        
        
        
def sample_data_dist(exp_setup,labels):
    len_dataset = labels.shape[0]
    if exp_setup == "class":
        
#         idx = []
#         online_iterations=int(len_dataset/batch_size)-1
#         samples_per_class=int(batch_size/args.n_classes)+1
#         selected_classes = random.sample(classes, args.n_classes)
        
#         for i in range(online_iterations):
#             class_index = [classes.index(elem) for elem in selected_classes]
            
#             for i, c in enumerate(class_index):
#                 class_indices = np.where(all_labels == c)[0]
#                 selected_indices = random.sample(list(class_indices), samples_per_class)
#                 idx.extend(selected_indices)
        selected_classes = random.sample(classes, args.n_classes)    
        class_index = [classes.index(elem) for elem in selected_classes]
        class_indices={}
        for c in class_index:
            class_indices[c]=np.where(labels == c)[0]

        samples_per_class = args.batch_size // args.n_classes #int(batch_size/args.n_classes)+1
        idx = []
        while True:
            this_batch=[]
            for label, indices in class_indices.items():
                if indices is not None:
                    this_batch.extend(indices[:samples_per_class])
                    class_indices[label]=class_indices[label][samples_per_class:]

            random.shuffle(this_batch)
            idx.extend(this_batch[:args.batch_size])
            
            
            if len(class_indices[label])<=args.batch_size:
                break
                
         
    elif exp_setup == "IS+CB":
        idx = [i for i in range(len_dataset)]
        random.shuffle(idx)
            
    elif exp_setup == "IS+CI":
        idx = dirichlet_split_noniid(np.array(labels), args.pi, 10)
        idx = np.concatenate(idx)
    elif exp_setup == "DS+CB":
        
        prob_per_class = []
        for cls_idx in range(number_of_classes):
            prob_per_class.append( args.rho ** (cls_idx / (number_of_classes - 1.0)) )
        prob_per_class = np.array(prob_per_class) / sum(prob_per_class)
        img_per_class = prob_per_class * len(labels)
        idx = []
        y_test_np = np.array(labels)
        for c, num in enumerate(img_per_class):
            all_of_c = np.where(y_test_np==c)[0]
            idx.append(np.random.choice(all_of_c, int(num)+1))
        idx = np.concatenate(idx)
        random.shuffle(idx)
        
    elif exp_setup == "DS+CI":
        
        prob_per_class = []
        for cls_idx in range(number_of_classes):
            prob_per_class.append( args.rho ** (cls_idx / (number_of_classes - 1.0)) )
        prob_per_class = np.array(prob_per_class) / sum(prob_per_class)
        img_per_class = prob_per_class * len(labels)
        idx = []
        y_test_np = np.array(labels)
        for c, num in enumerate(img_per_class):
            all_of_c = np.where(y_test_np==c)[0]
            idx.append(np.random.choice(all_of_c, int(num)+1))
        idx = np.concatenate(idx)
        idx2 = dirichlet_split_noniid(np.array([y_test_np[i] for i in idx]), args.pi, 10)
        idx = np.concatenate([idx[i] for i in idx2])
        
    return idx
        
def create_dataloader(images, labels, idx, batch_size,num_workers=2):
    image_shape=images[0].shape
    print(image_shape)
    selected_images = [images[i] for i in idx]
    selected_labels = np.array([labels[i] for i in idx])
    data = torch.tensor(selected_images)
    data = data.reshape(-1, image_shape[0],image_shape[1],image_shape[2])
    labels = torch.tensor(selected_labels)
    
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    dataset_trans = CustomTensorDataset(data=data, labels=labels, transform=transform_test)
    loader = DataLoader(dataset_trans, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
json_data = []

with open('/home/mila/p/paria.mehrbod/scratch/TTA/milaTTA/cifar_tentdelta.yaml', 'rb') as f:
    args_tent = yaml.safe_load(f.read())
config_obj = SimpleNamespace(**args_tent)
for seed in [0, 19, 22, 42, 81]:
    random.seed(seed)
    torch.manual_seed(seed)
#     name = '%s_%i_%i_Loop%i' % (corruption, severity, n_classes, seed)
    
    
    parameters["seed"]=seed
    parameters["corruption"]=corruption
    parameters["severity"]=severity
    

    if args.n_classes is not None:
        parameters["n_classes"]=args.n_classes
    
    print('==> Building model..')

    if model_arch == 'Resnet-26':
        base_model = ResNet26()

    net = base_model.to(device)

   
    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    if optimizer == 'SGD':
        opt = optim.SGD(net.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print("Model created.")

    trained_weights = torch.load( '/home/mila/p/paria.mehrbod/scratch/TTA/test_time_adaptation/utils/cifar10_Resnet-26_weights.pth')
    sd = trained_weights['net']
    net.load_state_dict(sd)
    
    net.eval()
    
    
    mask_name = '%s_%i' % (corruption, severity)
    print('Using masks of:', mask_name)
    spec_channels_path = '/home/mila/p/paria.mehrbod/scratch/TTA/test_time_adaptation/utils/all_files_with_scores.csv' # Using scores
    print('Importing file: ', spec_channels_path)
    spec_channels_file = glob.glob(spec_channels_path)
    scf = pd.read_csv(spec_channels_path)
    all_channels = scf[scoring].apply(ast.literal_eval)

    
    
    lame_model=LAME(copy.deepcopy(net),10,5,1)
#     delta_model= DELTA(config_obj,copy.deepcopy(net))
    threshold_choice = 'Linear-Channels'
    threshold = 0.1
#     delta_model= DELTA_Hybrid(config_obj,copy.deepcopy(net), threshold,  threshold_choice , dist_choice, switch, chosen_layer, all_channels,False)
#     delta_model_v2= DELTA_Hybrid(config_obj,copy.deepcopy(net), threshold,  threshold_choice , dist_choice, switch, chosen_layer, all_channels,True)
    delta_model = DELTA(config_obj,copy.deepcopy(net))
    threshold_choice = 'Linear-Channels'
    threshold = 0.1
    delta_model_v1= DELTA_Hybrid(config_obj,copy.deepcopy(net), threshold,  threshold_choice , dist_choice, switch, chosen_layer, all_channels,False)
    delta_model_v2= DELTA_Hybrid(config_obj,copy.deepcopy(net), threshold,  threshold_choice , dist_choice, switch, chosen_layer, all_channels,True)
    delta_model_v1_switch= DELTA_Hybrid(config_obj,copy.deepcopy(net), threshold,  threshold_choice , dist_choice,True, chosen_layer, all_channels,False)
    delta_model_v2_switch= DELTA_Hybrid(config_obj,copy.deepcopy(net), threshold,  threshold_choice , dist_choice, True, chosen_layer, all_channels,True)
    
    
    
    tentnet = setup_tent(copy.deepcopy(net),1,False)
    check_model(tentnet)    

#     base_model_stats = batch_norm_stats(net, print_stats=False)

    
    #First: Complete TTN
    threshold_choice = 'Fixed-Channels'
    threshold = 0
    norm_net_TTN = thc_specific.Threshold(copy.deepcopy(net), threshold, threshold_choice, dist_choice, switch, chosen_layer, all_channels, mask_name)

    #Second: Fixed 10% based on SOURCE channels
    threshold_choice = 'Fixed-Channels'
    threshold = 0.1
#     norm_net_Fixed = thc_specific.Threshold(copy.deepcopy(net), threshold, threshold_choice, dist_choice, switch, chosen_layer, all_channels, mask_name)
        
    #Third: Linearly adapting based on SOURCE channels
    threshold_choice = 'Linear-Channels'
    norm_net_linear = thc_specific.Threshold(copy.deepcopy(net), threshold, threshold_choice, dist_choice, switch, chosen_layer, all_channels, mask_name)

    #Fourth: Linearly adapting based on TARGET data
    # threshold_choice = 'Linear-Decay'
#     norm_net_target = thc.Threshold(copy.deepcopy(net), threshold, threshold_choice, dist_choice, switch, chosen_layer)
        
#     print('Models adapted for threshold.')

#     adapted_model_stats = batch_norm_stats(norm_net, print_stats=False)
#     adap_modules = list_modules(norm_net)

    #Data
    print('Importing dataset:')
    if severity > 0:
        images, labels = load_corrupt_data(corruption, severity)
    else:
        images, labels = load_normal_data()
    idx = sample_data_dist(args.exp_setup,labels)
#     dataset = [(image, label) for image, label in zip(images, labels)] 
    chosen_loader = create_dataloader(images, labels, idx,args.batch_size)
    
#     methods=["DELTA","NOT_ADAPTED","SourceLinear","TTN","TENT","LAME"]
#     methods=["DELTA","DELTA_v1","DELTA_v2","DELTA_switch_v1","DELTA_switch_v2"]
    methods = ["SourceLinear"]
#     adapt_model={"DELTA":delta_model,"DELTA_v1":delta_model_v1,"DELTA_v2":delta_model_v2,"DELTA_switch_v1":delta_model_v1_switch,"DELTA_switch_v2":delta_model_v2_switch}
    adapt_model={"SourceLinear":norm_net_linear}
#     adapt_model={"DELTA":delta_model,"TENT":tentnet,"NOT_ADAPTED":net,"LAME":lame_model,"SourceLinear":norm_net_linear,"TTN":norm_net_TTN}
    
    with torch.no_grad():
        
        correct = {}
        total = {}
        batch_acc = {}
        class_num = {}
        class_correct = {}
        class_avg_acc = {}
        cumulative_acc = {}
        for method in methods:
            correct[method] = 0
            total[method] = 0   
            batch_acc[method] = 0
            class_num[method] = np.array([0]*10)
            class_correct[method] = np.array([0]*10)
            class_avg_acc[method] = 0
            cumulative_acc[method] = 0
        results={}
        for batch_idx, (inputs, targets) in enumerate(chosen_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                batch_result={}
                
                for method in methods:
                    model=adapt_model[method]
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total[method] = targets.size(0)
                    correct[method] = predicted.eq(targets).sum().item()
                    batch_acc[method] = 100.*correct[method]/total[method]
                    for i, t in enumerate(targets):
                        class_num[method][t.item()] += 1
                        class_correct[method][t.item()] += (predicted[i]==t)
                    acc = (class_correct[method][class_num[method]!=0] / class_num[method][class_num[method]!=0])
                    class_avg_acc[method] = acc.mean() * 100.
                    cumulative_acc[method] = class_correct[method].sum() / class_num[method].sum() * 100
                    batch_result[f'{method}_cumulative_accuracy_{batch_idx}'] = cumulative_acc[method]
                    batch_result[f'{method}_batch_accuracy_{batch_idx}'] = batch_acc[method]
                    batch_result[f'{method}_class_accuracy_{batch_idx}'] = class_avg_acc[method]

                    results.update(batch_result)

        tentnet.reset()

        json_entry = {
        'parameters': parameters,
        'results': results
        }
        json_data.append(json_entry)

        
if not os.path.isfile(file):
    with open(file,"w+") as fp:
        json.dump([],fp)
with open(file,"r") as fp:
    listObj = json.load(fp)
listObj.extend(json_data)
with open(file, 'w') as json_file:
    json.dump(listObj, json_file, indent=4, separators=(',',': '))


import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from robustbench.data import load_cifar10c, load_cifar10
from test_utils import *
from models.DELTA_Res import ResNet, Bottleneck
import models.DELTA_Res as Resnet
from collections import Counter

import utils.thresholding_adaptingspecificchannels as thc_specific
import utils.thresholding_combined as thc
import tqdm
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
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Configs
'''
In this case, we used weights from Resnet-18 trained on Imagenet, 
and will test it on Imagenet_corrupted, 
with adapting using the channel-level information.
'''
torch.autograd.set_detect_anomaly(True)
#Fixed configs
model_arch = 'Resnet-50'
optimizer = 'SGD'
lr = 0.1
epochs = 200
weight_decay = 5e-4
momentum = 0.9
save_dir = '/home/mila/p/paria.mehrbod/scratch/TTA/test_time_adaptation/outputs/'
adaptation = 'norm_threshold'
dist_choice = 'Wasser'
chosen_layer = 52
tta_epochs = 1

switch = False
scoring = 'sum'

n_clients=10


#Model config
parser = argparse.ArgumentParser(description='PyTorch Byra Test-time Adaptation using Channel-level statistics')

#Dataset config
parser.add_argument('--corruption', default = 'gaussian_noise', type=str, help='there are 19 types of corruption. default is gaussian_noise')

parser.add_argument('--batch_size', default = 500, type=int, help='batch size')
parser.add_argument('--exp_setup', required= True ,choices=["DS+CB","DS+CI","IS+CI","IS+CB","class"], help='distribution of data')
parser.add_argument('--n_classes', type=int, help='number of classes, 1 to 10')
parser.add_argument('--rho', required=False, type=float, help='long tail factor')
parser.add_argument('--pi', required=False, type=float, help='dirichlet factor')
parser.add_argument('--severity', default = 0, type=int, help='severity, 0 to 5')
args = parser.parse_args()
corruption = args.corruption
severity = args.severity

number_of_classes = 1000



DATA_PATH='/home/mila/p/paria.mehrbod/scratch/TTA/data/Imagenet-C'
file= f"imagenet_online_b{args.batch_size}_sev{args.severity}_cor{args.corruption}"





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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),normalize])
    dataset = torchvision.datasets.ImageFolder("/home/mila/p/paria.mehrbod/scratch/TTA/data/Imagenet-C/{}/{}/".format(corruption, severity),transform=transform)
    dataset.targets = np.array(dataset.targets)
    dataset.samples = np.array(dataset.samples)
    return dataset.samples, dataset.targets

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
#     print(image_shape)
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


classes = range(1000)
json_data = []

for seed in  [0]: #2020 , 42, 81
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # name = '%s_%i_%i_Loop%i' % (corruption, severity, n_classes, seed)
#     selected_classes = random.sample(classes, n_classes)
    
    parameters["seed"]=seed
    parameters["corruption"]=corruption
    parameters["severity"]=severity

    if args.n_classes is not None:
        parameters["n_classes"]=args.n_classes
    
    print('==> Building model..')

    if model_arch == 'Resnet-50':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),normalize])
#         base_model = ResNet(Bottleneck, [3, 4, 6, 3])
        base_model = Resnet.__dict__["resnet50"](pretrained=True).cuda()
#         state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',progress=True)
#         base_model.load_state_dict(state_dict)
        
        
    
    if model_arch == 'Resnet-26':
        base_model = ResNet26()

    if model_arch == 'Resnet-18':    
        transform = ResNet18_Weights.IMAGENET1K_V1.transforms()   
        base_model=resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
    
    
    net = base_model.to(device)
    net.eval()
    
    mask_name = '%s_%i' % (corruption, severity)
    print('Using masks of:', mask_name)
    spec_channels_path = '/home/mila/p/paria.mehrbod/scratch/TTA/milaTTA/all_files_with_scores_sum.csv' # 
    scf = pd.read_csv(spec_channels_path)
    all_channels = scf[scoring].apply(ast.literal_eval)
    
    
    with open('/home/mila/p/paria.mehrbod/scratch/TTA/milaTTA/imagenetc_tentdelta.yaml', 'rb') as f:
        args_ = yaml.safe_load(f.read())
    config_obj = SimpleNamespace(**args_)
    
#     lame_model=LAME(net,10,5,1)
#     tentnet = setup_tent(copy.deepcopy(net),1,False)   
#     tentnet10 = setup_tent(copy.deepcopy(net),10,False)

    
    delta_model = DELTA(config_obj,copy.deepcopy(net))
    threshold_choice = 'Linear-Channels'
    threshold = 0.1
    delta_model_v1= DELTA_Hybrid(config_obj,copy.deepcopy(net), threshold,  threshold_choice , dist_choice, switch, chosen_layer, all_channels,False)
    delta_model_v2= DELTA_Hybrid(config_obj,copy.deepcopy(net), threshold,  threshold_choice , dist_choice, switch, chosen_layer, all_channels,True)
    delta_model_v1_switch= DELTA_Hybrid(config_obj,copy.deepcopy(net), threshold,  threshold_choice , dist_choice,True, chosen_layer, all_channels,False)
    delta_model_v2_switch= DELTA_Hybrid(config_obj,copy.deepcopy(net), threshold,  threshold_choice , dist_choice, True, chosen_layer, all_channels,True)
    
    
    
  
    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss()
    if optimizer == 'SGD':
        opt = optim.SGD(net.parameters(), lr=config_obj.optim_lr,
                          momentum=config_obj.optim_momentum, weight_decay=config_obj.optim_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print("Model created.")
    
    

#     base_model_stats = batch_norm_stats(net, print_stats=False)

    
    #First: Complete TTN
    threshold_choice = 'Fixed-Channels'
    threshold = 0
    norm_net_TTN = thc_specific.Threshold(copy.deepcopy(net), threshold,  'Fixed-Channels', dist_choice, switch, chosen_layer, all_channels, mask_name)

    #Second: Fixed 10% based on SOURCE channels
    threshold_choice = 'Fixed-Channels'
    threshold = 0.1
    norm_net_Fixed = thc_specific.Threshold(copy.deepcopy(net), threshold, threshold_choice, dist_choice, switch, chosen_layer, all_channels, mask_name)
        
    #Third: Linearly adapting based on SOURCE channels
    threshold_choice = 'Linear-Channels'
    norm_net_linear = thc_specific.Threshold(copy.deepcopy(net), threshold, threshold_choice, dist_choice, switch, chosen_layer, all_channels, mask_name)
    
    
    
    
    
    #Fourth: Linearly adapting based on TARGET data
#     threshold_choice = 'Linear-Decay'
#     norm_net_target = thc.Threshold(copy.deepcopy(net), threshold, threshold_choice, dist_choice, switch, chosen_layer)
    
#     #Fifth: Linearly adapting based on SOURCE -> but cheating by knowing the target class (aka Class-specific adaptation)
#     threshold_choice = 'Linear-Channels'
#     if selected_classes[0] == 0:
#         specific_channels_path = save_dir + 'investigating_masks_wasserthresh/investigating_Oct26/none_0_0_Linear-Decay.csv'
#     else:
#         specific_channels_path = save_dir + 'investigating_masks_wasserthresh/investigating_Oct26/*_' + str(selected_classes[0]) + '_*.csv' # Selecting automatically
#     print('Importing file (cheating): ', specific_channels_path)
#     specific_channels_file = glob.glob(specific_channels_path)
#     scf_spec = pd.read_csv(specific_channels_file[0])
#     specific_channels = scf_spec['Channels'].apply(ast.literal_eval)
#     norm_net_cheating = thc_specific.Threshold(copy.deepcopy(net), threshold, threshold_choice, dist_choice, switch, chosen_layer, specific_channels, mask_name)
        
#     print('Models adapted for threshold.')

#     adapted_model_stats = batch_norm_stats(norm_net, print_stats=False)
#     adap_modules = list_modules(norm_net)

    #Data
    print('Importing dataset:')
    if severity > 0:
#         print('Using %i classes, %s with severity %i' % (n_classes, corruption, severity))
        dataset = torchvision.datasets.ImageFolder("/home/mila/p/paria.mehrbod/scratch/TTA/data/Imagenet-C/{}/{}/".format(corruption, severity),
                                                   transform=transform)
        dataset.targets = np.array(dataset.targets)
    else:
#         print('Using %i classes, without corruptions' % (n_classes))
        path = "/network/datasets/imagenet.var/imagenet_torchvision/"
        dataset = torchvision.datasets.ImageFolder(path+'val/', transform=transform)
        dataset.targets = np.array(dataset.targets)
        print('Loaded dataset without corruptions.')

    indices =  sample_data_dist(args.exp_setup,dataset.targets)
    dataset.targets = dataset.targets[indices].tolist()
    dataset.samples = [dataset.samples[index] for index in indices]

    print('Loaded dataset with %i images.' % len(dataset))
    chosen_loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=2)

#     methods=["DELTA","DELTA_v1","DELTA_v2","DELTA_switch_v1","DELTA_switch_v2"]
#     adapt_model={"DELTA":delta_model,"DELTA_v1":delta_model_v1,"DELTA_v2":delta_model_v2,"DELTA_switch_v1":delta_model_v1_switch,"DELTA_switch_v2":delta_model_v2_switch}
#     adapt_model={"DELTA":delta_model,"DELTA_v2":delta_model_v2,"SourceLinear":norm_net_linear} #,"TENT":tentnet,"NOT_ADAPTED":net,"LAME":lame_model,,"TTN":norm_net_TTN
    methods=["SourceLinear"]
    adapt_model={"SourceLinear":norm_net_linear}
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
            class_num[method] = np.array([0]*number_of_classes)
            class_correct[method] = np.array([0]*number_of_classes)
            class_avg_acc[method] = 0
            cumulative_acc[method] = 0
        results={}
        for method in methods:
            print(method)
            model=adapt_model[method]
            batch_result={}
                
            for batch_idx, (inputs, targets) in enumerate(chosen_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
#                 print(predicted)
                for i, t in enumerate(targets):
                    class_num[method][t.item()] += 1
                    class_correct[method][t.item()] += (predicted[i]==t)
                if batch_idx%10==0:
                    print(batch_idx)
                    acc = (class_correct[method][class_num[method]!=0] / class_num[method][class_num[method]!=0])
                    class_avg_acc[method] = acc.mean() * 100.
                    cumulative_acc[method] = class_correct[method].sum() / class_num[method].sum() * 100
                    batch_result[f'{method}_cumulative_accuracy_{batch_idx}'] = cumulative_acc[method]
                    batch_result[f'{method}_class_accuracy_{batch_idx}'] = class_avg_acc[method]
                    total[method] = targets.size(0)
                    correct[method] = predicted.eq(targets).sum().item()
                    batch_acc[method] = 100.*correct[method]/total[method]
                    batch_result[f'{method}_batch_accuracy_{batch_idx}'] = batch_acc[method]

                    
                
#                     
            total[method] = targets.size(0)
            correct[method] = predicted.eq(targets).sum().item()
            batch_acc[method] = 100.*correct[method]/total[method]
#         for method in methods:
            acc = (class_correct[method][class_num[method]!=0] / class_num[method][class_num[method]!=0])
            class_avg_acc[method] = acc.mean() * 100.

            cumulative_acc[method] = class_correct[method].sum() / class_num[method].sum() * 100
            batch_result[f'{method}_cumulative_accuracy_{batch_idx}'] = cumulative_acc[method]
            batch_result[f'{method}_class_accuracy_{batch_idx}'] = class_avg_acc[method]
            batch_result[f'{method}_batch_accuracy_{batch_idx}'] = batch_acc[method]
#             print(batch_result)
            results.update(batch_result)

#         tentnet.reset()



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
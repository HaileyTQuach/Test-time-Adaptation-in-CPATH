import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter


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

import json
import pickle
from copy import deepcopy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Fixed configs
METHOD= "Adam"
WD= 0.


## LAME config
STEPS = 100
LR = 1e-3
BETA = 0.9
# OPTIMIZER = "Adam"
# BATCH_SIZE = 16 #x
# OPTIM_MOMENTUM =  0.9 #x
# DAMPENING = 0. 
# WEIGHT_DECAY = 0.0001 
# NESTEROV = True #x
AFFINITY="rbf"
FORCE_SYMMETRY=False
SIGMA=1
KNN=5


def ResNet26():
    return ResNet(Bottleneck, [2, 2, 2, 2], num_classes=10)


class AffinityMatrix:

    def __init__(self, **kwargs):
        pass

    def __call__(X, **kwargs):
        raise NotImplementedError

    def is_psd(self, mat):
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]
        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())


class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W


class rbf_affinity(AffinityMatrix):
    def __init__(self, sigma: float, **kwargs):
        self.sigma = sigma
        self.k = kwargs['knn']

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:,
                   -1]  # compute k^th distance for each point, [N, knn + 1]
        sigma = kth_dist.mean()
        rbf = torch.exp(- dist ** 2 / (2 * sigma ** 2))
        # mask = torch.eye(X.size(0)).to(X.device)
        # rbf = rbf * (1 - mask)
        return rbf


class linear_affinity(AffinityMatrix):

    def __call__(self, X: torch.Tensor):
        """
        X: [N, d]
        """
        return torch.matmul(X, X.t())


    
class LAME(nn.Module):
    """ Parameter-free Online Test-time Adaptation
    """
    def __init__(self, model, num_classes,steps,episodic):
        super().__init__()
        
        # configure model and optimizer
        self.model = model
        self.num_classes = num_classes
        self.steps=steps
        self.episodic=episodic
    
       
        self.configure_model()
        self.params, param_names = self.collect_params()
        if len(self.params) > 0:
            self.optimizer = self.setup_optimizer()
        else:
            print("here")
            self.optimizer = None
#         self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None
        self.print_amount_trainable_params()
        
#         # variables needed for single sample test-time adaptation (sstta) using a sliding window (buffer) approach
#         self.input_buffer = None
#         self.window_length = cfg.TEST.WINDOW_LENGTH
#         self.pointer = torch.tensor([0], dtype=torch.long).cuda()
        # sstta: if the model has no batchnorm layers, we do not need to forward the whole buffer when not performing any updates
        self.has_bn = any([isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()])

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()


        self.affinity = eval(f'{AFFINITY}_affinity')(sigma=SIGMA, knn=KNN)
        self.force_symmetry = FORCE_SYMMETRY

        # split up the model
#         self.feature_extractor, self.classifier = split_up_model(self.model, cfg.MODEL.ARCH, self.dataset_name)

        try:
            self.classifier = self.model.linear
            self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        except:
            self.classifier = self.model.fc
            self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-2])
            
        self.pool=nn.AdaptiveAvgPool2d((1, 1))
        
        self.model_state, _ = self.copy_model_and_optimizer()
       
    
    def print_amount_trainable_params(self):
        trainable = sum(p.numel() for p in self.params) if len(self.params) > 0 else 0
        total = sum(p.numel() for p in self.model.parameters())
        print(f"#Trainable/total parameters: {trainable}/{total} \t Fraction: {trainable / total * 100:.2f}% ")

    def setup_optimizer(self):
        if OPTIM_METHOD == 'Adam':
            return torch.optim.Adam(self.params,
                                    lr=LR,
                                    betas=(BETA, 0.999),
                                    weight_decay=WD)
        elif OPTIM_METHOD == 'SGD':
            return torch.optim.SGD(self.params,
                                   lr=LR,
                                   momentum=MOMENTUM,
                                   dampening=DAMPENING,
                                   weight_decay=WD,
                                   nesterov=NESTEROV)
        else:
            raise NotImplementedError

    
    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names
        
    def forward(self,x):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs    

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        
    @torch.no_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        imgs_test = x
        features = self.feature_extractor(imgs_test)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        outputs = self.classifier(features)

        # --- Get unary and terms and kernel ---
        unary = - torch.log(outputs.softmax(dim=1) + 1e-10)  # [N, K]

        features = F.normalize(features, p=2, dim=-1)  # [N, d]
        kernel = self.affinity(features)  # [N, N]
        if self.force_symmetry:
            kernel = 1 / 2 * (kernel + kernel.t())

        # --- Perform optim ---
        outputs = laplacian_optimization(unary, kernel)

        return outputs

    def configure_model(self):
        """Configure model"""
        self.model.eval()
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(self.model.state_dict())
        return model_state, None

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)

        

def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):

    E_list = []
    oldE = float('inf')
    Y = (-unary).softmax(-1)  # [N, K]
    for i in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
        exponent = -unary + pairwise
        Y = exponent.softmax(-1)
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        E_list.append(E)

        if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
            
#             print(f'Converged in {i} iterations')
            break
        else:
            oldE = E

    return Y


def entropy_energy(Y, unary, pairwise, bound_lambda):
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()
    return E


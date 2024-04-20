import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from PIL import Image, ImageOps
import staintools
from statistics import median
from tqdm import tqdm
import random
import argparse
import json

import copy 
from types import SimpleNamespace
import models.tent as tent
from models.delta import DELTA
from models.test_utils import LAME
import yaml
import torch.optim as optim

import models.sar as sar
from models.sam import SAM
import models.tent as TENT1
import math
import gc

def custom_transform(image_path):
    image = staintools.read_image(image_path)
    image = staintools.LuminosityStandardizer.standardize(image)
    normalized_image = stain_norm.transform(image)
    im_pil = Image.fromarray(normalized_image.astype('uint8'), 'RGB')  # Convert to PIL image for transforms
    transform = transforms.Compose([
            transforms.Resize((m_p_s, m_p_s)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform(im_pil)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.dataset = datasets.ImageFolder(root=self.root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.imgs[idx]
        if self.transform:
            img = self.transform(img_path)
        return img, label


def setup_tent(model,steps,episodic,METHOD,noaffine=False):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = TENT1.configure_model(model,noaffine=noaffine)
    params, param_names = TENT1.collect_params(model)
    if METHOD=="SGD":
        optimizer = optim.SGD(params,
                        lr=0.00025,
                        momentum=0.9
                        )
    elif METHOD=="Adam":
        optimizer = optim.Adam(params,
                    lr=1e-3,
                    betas=(0.9, 0.999),
                    weight_decay=0)

    tent_model = TENT1.Tent(model, optimizer,
                           steps=steps,#cfg.OPTIM.STEPS
                           episodic=episodic,
                          noaffine=noaffine)#cfg.MODEL.EPISODIC
    return tent_model


parser = argparse.ArgumentParser(description='pathology TTA')
parser.add_argument('--artifact', type=str)
parser.add_argument('--cor_path', type=str)
parser.add_argument('--batch_size',default=64, type=int)
parser.add_argument('--model_name', default = 'TvN_350_SN_D256_Initial_Ep7_fullmodel.pth', type=str)
parser.add_argument('--exp_type', default = 'corr_experiments', type=str)

args = parser.parse_args()
corrupt_data_path=args.cor_path
model_name = args.model_name
artifact = args.artifact
print(artifact)
print(os.getcwd())
st = staintools.read_image("./Artifact/15_stain_scheme/schemes_ready/standard_he_stain_small.jpg")
standardizer = staintools.LuminosityStandardizer.standardize(st)
stain_norm = staintools.StainNormalizer(method='macenko')
stain_norm.fit(st)


model_dir = 'Models'
m_p_s = 350



json_data=[]
for seed in  [0,19,22, 42, 81]: #2020 , 42, 81
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    with open('configs/cifar_tentdelta_adam.yaml', 'rb') as f:
        args_tent = yaml.safe_load(f.read())
    config_obj = SimpleNamespace(**args_tent)

    path_model = os.path.join(model_dir, model_name)
    net = torch.load(path_model) 
    net = net.cuda()
    net.eval()
    lame_model=LAME(copy.deepcopy(net),3,5,1)
    delta_model= DELTA(config_obj,copy.deepcopy(net))
    sarnet = sar.configure_model(copy.deepcopy(net))
    params, param_names = sar.collect_params(sarnet)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer, lr=0.00025, momentum=0.9) #lr suitable for batch size >32
    sar_model = sar.SAR(sarnet, optimizer, margin_e0=math.log(3)*0.40) # since we have 3 classes

    tentnet = setup_tent(copy.deepcopy(net),10,False,METHOD="Adam")
    TENT1.check_model(tentnet)

        
    methods=["TENT", "DELTA","NOT_ADAPTED","SAR","LAME"] #"TTN" "DELTA","NOT_ADAPTED", "LAME", "DELTA","NOT_ADAPTED", "LAME","SAR"
    adapt_model={"DELTA":delta_model,"TENT":tentnet,"NOT_ADAPTED":net,"LAME":lame_model,"SAR":sar_model} #"TTN":norm_net_TTN, 
    print("model name:", model_name, "artifact: ",artifact)
    custom_dataset = CustomDataset(root=f"Corrupted_data/{corrupt_data_path}/{artifact}/", transform=custom_transform)
    dataloader = DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=True)    
    
    
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
        class_num[method] = np.array([0]*3)
        class_correct[method] = np.array([0]*3)
        class_avg_acc[method] = 0
        cumulative_acc[method] = 0
    results={}
    for method in methods:
        batch_idx=0
        iters = iter(dataloader)
        while(batch_idx<10):
            print(batch_idx)
            try:
                inputs, targets = next(iters)
            except:
                print("empty mask")
                continue
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_result={}
            outputs=adapt_model[method](inputs)
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
            adapt_model["TENT"].reset()
            batch_idx+=1
            del inputs, targets, outputs
            torch.cuda.empty_cache()
            gc.collect()

    
    
    
    json_entry = {"parameters": {
            "seed":seed,
            "artifact": artifact,
            "model" : model_name
            },
            "results": results
            }
    json_data.append(json_entry)
        
        
path= f"TTA_on_corrupted/{args.exp_type}/{corrupt_data_path}/"
if not os.path.exists(path):
    os.makedirs(path)
with open(f"{path}/results_TTA_{artifact}_b{args.batch_size}_{model_name[:-4]}.json", 'w') as json_file:
    json.dump(json_data, json_file, indent=4, separators=(',',': '))

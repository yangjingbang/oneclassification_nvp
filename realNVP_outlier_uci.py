from realNVP import RealNVP
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torchvision.utils import save_image
import torch.utils.data as Data
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os
import math
import argparse
import pprint
import copy

result = []
batch_size_list = [256]
hidden_size_list = [10, 25, 50]
data_name_list = ['texture']  
iter_list = [2]
for data_name in data_name_list:
    for h_s in hidden_size_list:
        for b_s in batch_size_list:
            for idx, i in enumerate(iter_list):
                hidden_size = h_s
                normal_class = tuple(iter_list[0:idx + 1]) # 以哪一or几类的数据作为正类
                if data_name == 'texture':
                    dataset_raw = pd.read_table(r'%s.txt' % data_name, header=None, encoding='gb2312',
                                                     delim_whitespace=True).values
                else:
                    dataset_raw = np.loadtxt(r"%s.txt" % data_name, delimiter=',')
                batch_size = b_s
                X = dataset_raw[:,:-1]
                y = dataset_raw[:,-1]
                for i in range(y.shape[0]):
                    if y[i] not in normal_class:
                        y[i] = 0
                    else:
                        y[i] = 1
                y = y.astype(np.int16)
                X_train, X_test, label_train, label_test = train_test_split(X, y, test_size = 0.3, random_state=0)
                trainset = np.concatenate([X_train, label_train.reshape(-1,1)], axis = 1)

trainloader = Data.DataLoader(dataset=Data.TensorDataset(torch.from_numpy(X_train).float()),
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)

testset = np.concatenate([X_test, label_test.reshape(-1,1)], axis = 1)
testloader = Data.DataLoader(
                            dataset=Data.TensorDataset(torch.from_numpy(testset).float()),
                            batch_size=batch_size,
                            shuffle=True,)

idx_label_score = []
nu_list = [0.01, 0.1]
num_epochs = 200
n_blocks_list = [5,6]
hidden_size_list = [10, 25, 50]
eta = nn.Parameter(torch.tensor(0.))
#训练
for n_blocks in n_blocks_list:
    for hidden_size in hidden_size_list:
        for nu_cpu in nu_list:
            
            nu = torch.tensor(nu_cpu).float()
            model = RealNVP(n_blocks = n_blocks, input_size = X.shape[1], hidden_size= hidden_size, n_hidden=1,)
            
            #train
            opt = optim.Adam(list(model.parameters())+[eta,], lr=0.001)
            for epoch in range(num_epochs):
                model.train()
                for step, (inputs, ) in enumerate(trainloader):
                    log_px = model.log_prob(inputs)
                    loss = -(eta + 1 / (1 - nu) * torch.mean(F.relu(log_px - eta)))
                    opt.zero_grad()
                    loss.backward()
                    eta.grad*=-1
                    opt.step()
# 测试
#model.eval()
#Log_px = model.log_prob(torch.from_numpy(dataset[:, :-1]).float())
#eta = np.percentile(Log_px.data.numpy(), nu * 100)
testloader = Data.DataLoader(
    dataset=Data.TensorDataset(torch.from_numpy(testset).float()),
    batch_size=batch_size,
    shuffle=True,)

with torch.no_grad():
    for (data,) in testloader:
        inputs = data[:,:-1]
        labels = data[:,-1]
        outputs = model.log_prob(inputs) # 计算auc值与eta无关 但是可以作为在判断正负类的阈值
        scores = outputs
        # scores[scores<0] = 0
        # Save triples of (idx, label, score) in a list
        idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                    scores.cpu().data.numpy().tolist()))

# Compute AUC
labels, scores = zip(*idx_label_score)
labels = np.array(labels)
scores = np.array(scores)
auc = roc_auc_score(labels, scores)
print("Test set AUC: {:.2f}%, nu= {}, normal_class = {}, hidden_size = {}".format(
    100. * auc, nu, normal_class, hidden_size))
result.append([auc, nu, normal_class, batch_size, hidden_size, data_name])
pd_res = pd.DataFrame(result, columns=['auc', 'nu', 'normal_class', 'batch_size', 'hidden_size', 'data_name'])
pd_res.to_csv('outlier_uci_RealNVP.csv')
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

def make_annulus(sample_num = 2000, R_l = 1, R_u = 2):
    t = np.random.rand(sample_num) * 2 * np.pi - np.pi # -pi < t < pi, 表示角度
    r = np.sqrt(np.random.rand(sample_num)*(R_u*R_u-R_l*R_l)+R_l*R_l)
    dataset = np.empty([sample_num,2])
    dataset[:,0] = r*np.cos(t)
    dataset[:,1] = r*np.sin(t)

    return dataset

batch_size = 256
rng = np.random.RandomState(42)

X_in = make_moons(n_samples=2000, noise=0.05, random_state=0)[0]
#X_in = make_annulus()
#X_in = make_blobs(n_samples = 2000, n_features = 2, centers=[[2, 2], [-2, -2]],cluster_std=[0.5, 0.5])[0]
Y_in = np.array([1]*2000)
X_out = rng.uniform(low=-4, high=4, size=(200, 2))
Y_out = np.array([0]*200)

X = np.concatenate([X_in, X_out], axis=0)
Y = np.concatenate([Y_in, Y_out], axis=0)
X_train, X_test, train_label, test_label = train_test_split(X, Y, test_size = 0.3, random_state=0)

loader = Data.DataLoader(
    dataset=Data.TensorDataset(torch.from_numpy(X_train).float().cuda()),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

test_label=test_label.reshape(-1,1)
testset = np.concatenate([X_test, test_label], axis=1)
testloader = Data.DataLoader(
    dataset=Data.TensorDataset(torch.from_numpy(testset).float().cuda()),
    batch_size=batch_size,
    shuffle=True,
)

num_epochs = 300
batch_size = 256
eta = nn.Parameter(torch.tensor(-4.0).cuda())
nu = 0.1

model = RealNVP(n_blocks = 5, input_size = X.shape[1], hidden_size= 80, n_hidden=1,).cuda()
opt = optim.Adam(list(model.parameters())+[eta,], lr=0.001)
for epoch in range(num_epochs):
    model.train()
    for step, (inputs, ) in enumerate(loader):
        log_px = model.log_prob(inputs)
        loss = -(eta + 1 / (1 - nu) * torch.mean(F.relu(log_px - eta)))
        opt.zero_grad()
        loss.backward()
        eta.grad*=-1
        opt.step()

colors = np.array(['#377eb8', '#ff7f00'])
model.eval()
XX, YY = np.meshgrid(np.linspace(-4,4,100), np.linspace(-4, 4, 100))
X1 = torch.from_numpy(XX.reshape(-1,1)).float()
Y1 = torch.from_numpy(YY.reshape(-1,1)).float()
XY = torch.cat((X1, Y1), 1)
XY=XY.float().cuda()
Z = model.log_prob(XY)
#thre = np.asscalar(eta.cpu().data.numpy())
thre=eta
print(thre)
ZZ = Z.cpu().data.numpy().reshape(XX.shape)
plt.figure()
#plt.plot(data_set[:,0],data_set[:,1],'r.')
plt.scatter(X_in[:, 0], X_in[:, 1], s=10, color=colors[1])
plt.scatter(X_out[:, 0], X_out[:, 1], s=10, color=colors[0])
plt.contour(XX, YY, ZZ, levels=[thre,], linewidths=2, colors='black')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

model.eval()
idx_label_score = []
with torch.no_grad():
    for (data,) in testloader:
        inputs = data[:,:2]
        labels = data[:,2]
        outputs = model.log_prob(inputs)
        scores = outputs

        # Save triples of (idx, label, score) in a list
        idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                    scores.cpu().data.numpy().tolist()))

test_scores = idx_label_score

# Compute AUC
labels, scores = zip(*idx_label_score)
labels = np.array(labels)
scores = np.array(scores)

test_auc = roc_auc_score(labels, scores)
print('moons:   Test set AUC: {:.2f}%'.format(100. * test_auc))
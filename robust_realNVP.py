import numpy as np
import torch
import torchvision.datasets as dst
import torch.utils.data as Data
from realNVP import RealNVP
import torch.optim as optim
from torch.utils.data import Subset
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from datasets.main import load_dataset

dataset = load_dataset(dataset_name = 'mnist', data_path='./data', normal_class=(2))
#dataset = load_dataset(dataset_name = 'cifar10', data_path='./data', normal_class=(1,2))
train_loader = Data.DataLoader(dataset = dataset.train_set, batch_size=200, shuffle=False, drop_last=True)
test_loader = Data.DataLoader(dataset=dataset.test_set, batch_size=200, shuffle=False, drop_last=True)

if __name__ == "__main__":
    eta_set = [0., 5., 10.]
    hidden_size_set = [50,60,80,100]
    num_epochs_set = [100,150,200]
    n_blocks_set = [5,6]
    nu = 0.1
    for hidden_size in hidden_size_set:
        for num_epochs in num_epochs_set:
            for n_blocks in n_blocks_set:
                for eta in eta_set: 
                    model = RealNVP(n_blocks = 5, input_size = 28*28*1, hidden_size=50, n_hidden=1,)
                    # model = RealNVP(n_blocks = 5, input_size = 32*32*3, hidden_size=100, n_hidden=1,)

                    #eta 根据数据集不同需要改变
                    eta = nn.Parameter(torch.tensor(eta))
                    # =====================train=====================

                    opt = optim.Adam(list(model.parameters())+[eta,], lr=0.001)
                    for epoch in range(num_epochs):
                        model.train()
                        for data in train_loader:
                            inputs, _, _ = data
                            log_px = model.log_prob(inputs)
                            loss = -(eta + 1 / (1 - nu) * torch.mean(F.relu(log_px - eta)))
                            opt.zero_grad()
                            loss.backward()
                            eta.grad*=-1
                            opt.step()
                    

                # =====================test=======================
                idx_label_score = []
                model.eval()
                with torch.no_grad():
                    for data in test_loader:
                        inputs, labels, _ = data
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
                print("hidden_size={}, n_blocks={}, eta={}, num_epochs={}\n".format(hidden_size,n_blocks,eta,num_epochs))
                print('Test set AUC: {:.2f}%\n\n'.format(100. * test_auc))
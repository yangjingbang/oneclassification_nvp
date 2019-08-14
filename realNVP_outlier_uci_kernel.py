from realNVP import RealNVP

import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_uci_data(file_name, normal_class, batch_size=256, scale='minmax'):
    if file_name == 'texture':
        dataset_raw = pd.read_table(r'texture.txt', header=None, encoding='gb2312',
                                    delim_whitespace=True).values
    else:
        dataset_raw = np.loadtxt(r"{}.txt".format(file_name), delimiter=',')
    for i in range(dataset_raw.shape[0]):
        if dataset_raw[i, -1] not in normal_class:
            dataset_raw[i, -1] = 0
        else:
            dataset_raw[i, -1] = 1
    inliners = dataset_raw[dataset_raw[:, -1] == 1]
    outliners_all = dataset_raw[dataset_raw[:, -1] == 0]
    outliners_num = int(0.1*inliners.shape[0])
    idx = np.random.randint(outliners_all.shape[0], size=outliners_num)
    outliners = outliners_all[idx, :]
    all_data = np.concatenate([inliners, outliners], axis=0)
    X_train, X_test, label_train, label_test = train_test_split(
        all_data[:, :-1], all_data[:, -1], test_size=0.3, random_state=0)
    # Scale Data
    scaler = MinMaxScaler() if scale == 'minmax' else StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    trainset = np.concatenate([X_train, label_train.reshape(-1, 1)], axis=1)
    trainloader = Data.DataLoader(dataset=Data.TensorDataset(torch.from_numpy(X_train).float()),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    testset = np.concatenate([X_test, label_test.reshape(-1,1)], axis =1)
    testloader = Data.DataLoader(dataset=Data.TensorDataset(torch.from_numpy(testset).float()),
                                 batch_size=batch_size,
                                 shuffle=True,)
    return trainset, testset, trainloader, testloader


def robust_realnvp(data, n_blocks=5, hidden_size=25):
    #  Setting
    nu = 0.1
    num_epochs = 400
    trainset, testset, trainloader, testloader = data
    #  Training
    i = 0
    eta = nn.Parameter(torch.tensor(-4.))
    model = RealNVP(n_blocks=n_blocks, input_size=trainset.shape[1]-1, hidden_size=hidden_size, n_hidden=1, )
    opt_1 = optim.Adam(list(model.parameters()), lr=0.001)
    opt_2 = optim.Adam([eta,], lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        if i == 0:
            for step, (inputs, ) in enumerate(trainloader):
                log_px = model.log_prob(inputs)
                loss_1 = -torch.mean(log_px)
                opt_1.zero_grad()
                loss_1.backward()
                opt_1.step()
        else:
            for step, (inputs, ) in enumerate(trainloader):
                log_px = model.log_prob(inputs)
                loss_1 = -(eta + 1 / (1 - nu) * torch.mean(F.relu(log_px - eta)))
                opt_1.zero_grad()
                loss_1.backward(retain_graph=True)
                opt_1.step()

                loss_2 = -loss_1
                opt_2.zero_grad()
                loss_2.backward()
                opt_2.step()

        model.eval()
        out = model.log_prob(torch.from_numpy(trainset[:, :-1]).float())
        eta = np.percentile(out.cpu().data.numpy(), 100*nu)
        eta = nn.Parameter(torch.tensor(eta))
        if i % 10 == 0:
            print('epoch=', i, loss_1)
        i += 1

    # Calculate AUC
    model.eval()
    idx_label_score = []
    with torch.no_grad():
        for (data,) in testloader:
            inputs = data[:, :-1]
            labels = data[:, -1]
            outputs = model.log_prob(inputs)
            scores = outputs
            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))
    labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)
    print('Robust RealNVP \t Test set AUC: {:.2f}%'.format(100. * test_auc))
    return test_auc


def non_deep_methods(data):
    outliers_fraction = 0.1
    trainset, testset, trainloader, testloader = data
    anomaly_algorithms = [
        ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
        ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                          gamma=0.1)),
        ("Isolation Forest", IsolationForest(behaviour='new',
                                             contamination=outliers_fraction,
                                             random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor(
            n_neighbors=35, contamination=outliers_fraction, novelty=True))]
    for name, algorithm in anomaly_algorithms:
        algorithm.fit(trainset[:, :-1])
        pred_scores = algorithm.score_samples(testset[:, :-1])
        test_auc = roc_auc_score(testset[:, -1], pred_scores)
        print('Algorithm: {} \t Test set AUC: {:.2f}%'.format(name, 100. * test_auc))


def search_parameters(data):
    results = []
    blocks_list = [4, 8]
    hidden_size_list = [10, 50, 100]
    for n_blocks in blocks_list:
        for hidden_size in hidden_size_list:
            auc = robust_realnvp(data, n_blocks=n_blocks, hidden_size=hidden_size)
            results.append([n_blocks, hidden_size, auc])
            print(n_blocks, hidden_size, auc)
    res_pd = pd.DataFrame(results, columns=['n_blocks', 'hidden_size', 'auc'])
    print(res_pd)


def robust_svm(data, M=50):
    """
    :param data:
    :param kernel: input kernel(x, y) and return a real value
    :param M: Representative points, range from 10 to 10^2
    :return:
    """

    #  Setting
    nu = 0.1
    num_epochs = 10
    trainset, testset, trainloader, testloader = data
    idx = np.random.randint(0, trainset.shape[0], M)
    Y = trainset[idx, :-1]
    #  Training
    i = 0
    eta = nn.Parameter(torch.tensor(-4.).to(torch.float32))
    c_i = nn.Parameter(torch.tensor([1 / M]*M).to(torch.float32))
    c_j = nn.Parameter(torch.tensor([1 / M]*M).to(torch.float32))
    y = nn.Parameter(torch.tensor(Y).to(torch.float32))
    opt_1 = optim.Adam([c_i, c_j, y], lr=0.002)
    opt_2 = optim.Adam([eta, ], lr=0.002)

    for epoch in range(num_epochs):
        for step, (inputs,) in enumerate(trainloader):
            fx = svm_fx(kernel, c_i, c_j, inputs.to(torch.float32), M, y)
            obj_fun_1 = -(eta + 1 / (1 - nu) * torch.mean(F.relu(fx - eta)))
            opt_1.zero_grad()
            obj_fun_1.backward(retain_graph=True)
            opt_1.step()

            obj_fun_2 = -obj_fun_1
            opt_2.zero_grad()
            obj_fun_2.backward()
            opt_2.step()

        out = svm_fx(kernel, c_i, c_j, torch.from_numpy(trainset[:, :-1]).to(torch.float32), M, y)
        eta = np.percentile(out.cpu().data.numpy(), 100*nu)
        eta = nn.Parameter(torch.tensor(eta).to(torch.float32))
        if i % 10 == 0:
            print('epoch=', i, obj_fun_1)
        i += 1

    # Calculate AUC
    idx_label_score = []
    with torch.no_grad():
        for (data,) in testloader:
            inputs = data[:, :-1]
            labels = data[:, -1]
            outputs = svm_fx(kernel, c_i, c_j, inputs.to(torch.float32), M, y)
            scores = outputs
            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))
    labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)
    print('Robust RealNVP \t Test set AUC: {:.2f}%'.format(100. * test_auc))


def svm_fx(kernel, c_i, c_j, inputs, M, y):
    """
    this function can generative the consin distance between w and \pi (x)
    :param kernel:
    :param c_i:
    :param c_j:
    :param inputs:
    :param M:
    :param y:
    :return:
    """
    def fx_in(inputs):
        fun_1 = lambda x: torch.sum(c_i[x] * kernel(inputs, y[x]))
        p_1 = sum(list(map(fun_1, range(M))))

        fun_2 = lambda x: torch.sum(c_i.mul(c_j[x] * kernel(y, y[x])))
        p_2 = sum(list(map(fun_2, range(M))))
        fx = -0.5 + p_1 - p_2  # Because C_i K(x_i, y_i) C_j^T is symmetric, so it can directly multiply 2.
        return fx
    res = torch.tensor(list(map(lambda x: fx_in(inputs[x]), range(inputs.shape[0]))))
    return res


def kernel(x, y, kernel_name='gaussian'):
    if kernel_name == 'gaussian':
        sigma = 0.1
        return torch.exp(-0.5*torch.norm(x-y)**2 / sigma**2)


if __name__ == '__main__':
    data_name = 'texture'
    normal_class = (2, ) if data_name == 'texture' else (1,)
    data = get_uci_data(data_name, normal_class=normal_class)  # if 'texture', normal_class = (2,), otherwise = (1,)
    # non_deep_methods(data)
    # robust_realnvp(data, n_blocks=5, hidden_size=5)
    # search_parameters(data)
    robust_svm(data)
    print('Dataset: {}'.format(data_name))





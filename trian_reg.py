from realNVP import RealNVP

from PIL import Image
import random
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
import torch
import torchvision.transforms as transforms


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
    y = nn.Parameter(torch.tensor(Y).to(torch.float32))
    opt_1 = optim.Adam([c_i, y], lr=0.002)
    opt_2 = optim.Adam([eta, ], lr=0.002)

    for epoch in range(num_epochs):
        for step, (inputs,) in enumerate(trainloader):
            fx = svm_fx(kernel, c_i, inputs.to(torch.float32), M, y)
            obj_fun_1 = -(eta + 1 / (1 - nu) * torch.mean(F.relu(fx - eta)))
            opt_1.zero_grad()
            obj_fun_1.backward(retain_graph=True)
            opt_1.step()

            obj_fun_2 = -obj_fun_1
            opt_2.zero_grad()
            obj_fun_2.backward()
            opt_2.step()

        out = svm_fx(kernel, c_i, torch.from_numpy(trainset[:, :-1]).to(torch.float32), M, y)
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
            outputs = svm_fx(kernel, c_i, inputs.to(torch.float32), M, y)
            scores = outputs
            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))
    labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)
    print('Robust RealNVP \t Test set AUC: {:.2f}%'.format(100. * test_auc))


def svm_fx(kernel, c_i, inputs, M, y):
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

        fun_2 = lambda x: torch.sum(c_i.mul(c_i[x] * kernel(y, y[x])))
        p_2 = sum(list(map(fun_2, range(M))))
        fx = -0.5 + p_1 - 0.5 * p_2
        # Because C_i K(x_i, y_i) C_j^T is symmetric, so it can directly multiply 2.
        return fx
    res = torch.tensor(list(map(lambda x: fx_in(inputs[x]), range(inputs.shape[0]))))
    return res


def svm_fx_vec(kernel, c_i, inputs, M, y):
    # aim to make the svm_fx vectorized to speed calculation
    pass


def kernel(x, y, kernel_name='gaussian'):
    if kernel_name == 'gaussian':
        sigma = 0.1
        return torch.exp(-0.5*torch.norm(x-y)**2 / sigma**2)


def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x

class BaseADDataset(ABC):
    """Anomaly detection dataset base class."""

    def __init__(self, root: str):
        super().__init__()
        self.root = root  # root path to data

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__


class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        return train_loader, test_loader


class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=(1), anomaly_ratio=0.1):
        super().__init__(root)

        '''
        self.n_classes = 2 
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        '''

        self.n_classes = 2
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))

        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]],
                                                             [min_max[normal_class][1] - min_max[normal_class][0]])])
        '''
        a1 = []
        a2 = []
        for i in self.normal_classes:
            a1.append(min_max[i][0])
            a2.append(min_max[i][1] - min_max[i][0])
        a1 = list([sum(a1)/len(a1)])
        a2 = list([sum(a2)/len(a2)])

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize(a1, a2)])
        '''
        target_transform = transforms.Lambda(lambda x: int(x in self.normal_classes))

        train_set = MyMNIST(root=self.root, train=True, download=True,
                            transform=transform, target_transform=target_transform)
        # pdb.set_trace()
        # Subset train_set to normal class
        train_idx_normal = list(
            get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes))
        train_idx_outlier = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(),
                                                 self.outlier_classes)
        train_idx_outlier = random.sample(train_idx_outlier, int(len(train_idx_normal)*anomaly_ratio))
        train_idx_normal.extend(train_idx_outlier)
        idx_all = np.array(train_idx_normal)
        # 70% idx_all splited as train 30% as test
        # permutation can mix idx_all so we can sampele
        shuffled_indices = np.random.permutation(idx_all)
        train_set_idx = int(len(shuffled_indices) * 0.6)
        validation_set_idx = int(len(shuffled_indices) * 0.8)
        train_indices = shuffled_indices[:train_set_idx]
        validation_indices = shuffled_indices[train_set_idx:validation_set_idx]
        test_indices = shuffled_indices[validation_set_idx:]
        self.train_set = Subset(train_set, train_indices)
        self.test_set = Subset(train_set, test_indices)
        self.validation_Set = Subset(train_set, validation_indices)


class MyMNIST(MNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed


class MyCIFAR10(CIFAR10):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyCIFAR10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed


class CIFAR10_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=(1, 2)):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = normal_class
        self.outlier_classes = list(range(0, 10))
        for i in self.normal_classes:
            self.outlier_classes.remove(i)

        # self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]
        '''
        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]] * 3,
                                                             [min_max[normal_class][1] - min_max[normal_class][0]] * 3)])
        '''

        a1 = []
        a2 = []
        for i in self.normal_classes:
            a1.append(min_max[i][0])
            a2.append(min_max[i][1] - min_max[i][0])
        a1 = list([sum(a1) / len(a1)])
        a2 = list([sum(a2) / len(a2)])

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize(a1, a2)])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyCIFAR10(root=self.root, train=True, download=True,
                              transform=transform, target_transform=target_transform)
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(torch.tensor(train_set.targets).clone().data.cpu().numpy(),
                                                self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyCIFAR10(root=self.root, train=False, download=True,
                                  transform=transform, target_transform=target_transform)


def load_dataset(dataset_name, data_path, normal_class, anomaly_ratio):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class, anomaly_ratio=anomaly_ratio)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    return dataset


def mnist_data(normal_class, anomaly_ratio, batch_size=256):
    dataset = load_dataset(dataset_name='mnist', data_path='./data', normal_class=(normal_class),
                           anomaly_ratio=anomaly_ratio)
    # dataset = load_dataset(dataset_name = 'cifar10', data_path='./data', normal_class=(1,2))
    train_loader = Data.DataLoader(dataset=dataset.train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = Data.DataLoader(dataset=dataset.validation_Set, batch_size=256, shuffle=False, drop_last=False)
    test_loader = Data.DataLoader(dataset=dataset.test_set, batch_size=256, shuffle=False, drop_last=False)
    return train_loader, test_loader, validation_loader


def realnvp_mnist(hidden_size, n_blocks, anomaly_ratio, re_coefficient, normal_class, batch_size, num_epochs):
    # Data setting
    data = mnist_data(normal_class=normal_class,
                      anomaly_ratio=anomaly_ratio,
                      batch_size=batch_size)

    # Model setting
    train_loader, test_loader, validation_loader = data
    nu = anomaly_ratio
    model = RealNVP(n_blocks=n_blocks, input_size=28 * 28 * 1, hidden_size=hidden_size,
                    n_hidden=1, )
    model = model.cuda()
    # training
    initial_opt = optim.Adam(list(model.parameters()), lr=0.001, weight_decay=1e-6)
    eta = nn.Parameter(torch.tensor(1.0).cuda())
    opt = optim.Adam(list(model.parameters()) + [eta, ], lr=0.001, weight_decay=1e-6)
    # model, epoch = train_mnist_earlystopping(model, train_loader, validation_loader, nu,
    #                                          eta, opt, num_epochs, initial_opt)
    model, epoch = train_mnist_regulariation(model, train_loader, test_loader, nu, re_coefficient, eta, opt, num_epochs, initial_opt)

    test_auc, labels, scores = calculate_auc(test_loader, model)
    return test_auc, scores, labels, epoch


def train_mnist_earlystopping(model, train_loader, validation_loader, nu, eta, opt, num_epochs, initial_opt):
    early_stop = EarlyStopping(patience=2)
    for epoch in range(num_epochs):
        model.train()
        if epoch > 0:
            #
            for data in train_loader:
                inputs, _, _ = data
                log_px = model.log_prob(inputs.cuda())
                loss = -(eta + 1 / (1 - nu) * torch.mean(F.relu(log_px - eta)))
                opt.zero_grad()
                loss.backward()
                eta.grad *= -1
                opt.step()
                # eta_change
                model.eval()
                out = batch_ouput(train_loader, model)
                eta = np.percentile(out, 100 * nu)
                eta = nn.Parameter(torch.tensor(eta).cuda())
        else:
            # Only maximize likelihood of train data without eta when epoch = 0
            for data in train_loader:
                inputs, _, _ = data
                log_px = torch.mean(model.log_prob(inputs.cuda()))
                loss = -log_px
                initial_opt.zero_grad()
                loss.backward()
                initial_opt.step()

            # Calculate eta
            model.eval()
            out = batch_ouput(train_loader, model)
            eta = np.percentile(out, 100 * nu)
            eta = nn.Parameter(torch.tensor(eta).cuda())

        # Early stopping
        val_auc, _, _ = calculate_auc(validation_loader, model)
        early_stop(val_auc, model)
        if early_stop.early_stop:
            break
        print('epoch:{}, loss:{}'.format(epoch, loss))
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model, epoch


def train_mnist_regulariation(model, train_loader, test_loader, nu, re_coefficient, eta, opt, num_epochs, initial_opt):
    for epoch in range(num_epochs):
        model.train()
        if epoch > 0:
            for data in train_loader:
                inputs, _, _ = data
                log_px = model.log_prob(inputs.cuda())
                regularization_term = re_coefficient * realnvp_regularization(model, 'E_2', inputs)
                loss = -(eta + 1 / (1 - nu) * torch.mean(F.relu(log_px - eta)) - regularization_term)
                opt.zero_grad()
                loss.backward()
                eta.grad *= -1
                opt.step()
                # eta_change
                model.eval()
                out = batch_ouput(train_loader, model)
                eta = np.percentile(out, 100 * nu)
                eta = nn.Parameter(torch.tensor(eta).cuda())
        else:
            # Only maximize likelihood of train data without eta when epoch = 0
            for data in train_loader:
                inputs, _, _ = data
                log_px = torch.mean(model.log_prob(inputs.cuda()))
                loss = -log_px
                initial_opt.zero_grad()
                loss.backward()
                initial_opt.step()
                # try to have a initial train for one batch
                break

            # Calculate eta
            model.eval()
            out = batch_ouput(train_loader, model)
            eta = np.percentile(out, 100 * nu)
            eta = nn.Parameter(torch.tensor(eta).cuda())
        test_auc, _, _ = calculate_auc(test_loader, model)
        print(f'epoch = {epoch}, loss = {loss:.2f}, test auc = {test_auc:.3f} regularization lambda = {re_coefficient}')
    return model, epoch


def realnvp_regularization(model, term, inputs):
    batch_size = inputs.shape[0]
    if term == 'E_F':
        z = model.base_dist.sample((batch_size, 1)).squeeze()
        samples, sum_log_abs_det_jacobians = model.inverse(z)
        log_prob = torch.mean(torch.sum(model.base_dist.log_prob(z) - sum_log_abs_det_jacobians, dim=1))
        return log_prob
    else:
        z = model.base_dist.sample((batch_size, 1)).squeeze()
        z.requires_grad = True
        samples, sum_log_abs_det_jacobians = model.inverse(z)
        log_prob = torch.mean(torch.sum(model.base_dist.log_prob(z) - sum_log_abs_det_jacobians, dim=1))
        log_prob.backward(retain_graph=True)
        dz_dx = torch.exp(-sum_log_abs_det_jacobians)
        out = dz_dx * z.grad
        if term == 'E_1':
            return torch.mean(torch.norm(out, p=1, dim=1, keepdim=True))
        elif term == 'E_2':
            return torch.mean(torch.norm(out, p=2, dim=1, keepdim=True))
        elif term == 'E_2_2':
            return torch.mean(torch.norm(out, p=2, dim=1, keepdim=True) ** 2)
        else:
            raise ValueError('Unexpected regularization term')


def batch_ouput(loader, model):
    indices = loader.dataset.indices
    res = []
    qc = int(len(indices) / 20)
    for i in range(20):
        sample_idx = indices[qc*i:qc*(i+1)]
        all_train = loader.dataset.dataset.train_data[sample_idx].float()
        out = model.log_prob(all_train.cuda())
        res.extend(out.cpu().data.numpy().tolist())
    return res


def calculate_auc(test_loader, model):
    idx_label_score = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels, _ = data
            outputs = model.log_prob(inputs.cuda())
            scores = outputs
            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))
    # Compute AUC
    labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)
    return test_auc, labels, scores


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, if_print=True):
        """
        :param patience: How long to wait after last time validation loss improved
        """
        self.patience = patience
        self.counter = 0
        self.best_auc = None
        self.early_stop = False
        self.best_auc = 0
        self.if_print = if_print

    def __call__(self, val_auc, model):
        if self.best_auc is None:
            self.best_auc = val_auc
            self.save_checkpoint(val_auc, model)
        elif val_auc < self.best_auc:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        else:
            self.best_auc = val_auc
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, auc, model):
        """
        Save model when validation auc increases
        """
        torch.save(model.state_dict(), 'checkpoint.pt')
        if self.if_print:
            print(f'best auc increases ({self.best_auc:.4f} --> {auc:.4f}).  Saving model')
        self.best_auc = auc

def tune_realnvp_mnist():
    res = []
    num_epochs_list = [40]
    anomaly_ratio = 0.1
    batch_size = 256
    n_blocks_list = [8]
    normal_class_list = [0]  #  range(10)
    hidden_size_list = [100]
    re_coefficient = 0.5
    for num_epochs in num_epochs_list:
        for normal_class in normal_class_list:
            for n_blocks in n_blocks_list:
                for hidden_size in hidden_size_list:
                    auc, scores, labels, trained_epoch = realnvp_mnist(
                                                        hidden_size=hidden_size,
                                                        n_blocks=n_blocks,
                                                        anomaly_ratio=anomaly_ratio,
                                                        num_epochs=num_epochs,
                                                        re_coefficient=re_coefficient,
                                                        normal_class=normal_class,
                                                        batch_size=batch_size,
                                                        )
                    print(f"hidden_size={hidden_size}   normal class={normal_class}   epochs={num_epochs} "
                          f"anomaly ratio={anomaly_ratio} n_blocks={n_blocks}  auc={100*auc:.2f}%   "
                          f" trained_epoch={trained_epoch}")

                    res.append([hidden_size,
                                normal_class,
                                num_epochs,
                                anomaly_ratio,
                                n_blocks, auc, trained_epoch])
                    # pd.DataFrame(scores, columns=['scores']).to_excel('scores.xlsx')
                    # pd.DataFrame(labels, columns=['labels']).to_excel('labels.xlsx')


    pd_res = pd.DataFrame(res, columns=['hidden_size', 'normal class',
                                        'epochs', 'anomaly ratio', 'n_blocks', 'auc', 'trained_epoch'])
    pd_res.to_csv('mnist_robust_realnvp.csv')


def non_deep_mnist(data, normal_class):
    from sklearn.model_selection import GridSearchCV

    outliers_fraction = 0.1
    train_loader, test_loader = data
    res = []
    anomaly_algorithms = [
        # ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
        ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf")),
        ("Isolation Forest", IsolationForest(behaviour='new',
                                             contamination=outliers_fraction,
                                             random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor(
            n_neighbors=35, contamination=outliers_fraction, novelty=True))]
    for name, algorithm in anomaly_algorithms:
        # fit data
        inputs = train_loader.dataset.dataset.train_data[train_loader.dataset.indices].float()
        pca = PCA(n_components=400)
        train_data = pca.fit_transform(np.reshape(inputs, (inputs.shape[0], -1)))
        algorithm.fit(train_data)
        # test auc
        test_set = test_loader.dataset.dataset.train_data[test_loader.dataset.indices].float()
        pca_test_set = pca.fit_transform(np.reshape(test_set, (test_set.shape[0], -1)))
        pred_scores = algorithm.score_samples(pca_test_set)
        original_labels = test_loader.dataset.dataset.train_labels[test_loader.dataset.indices]
        labels = [1 if original_labels[i] == normal_class else 0 for i in range(original_labels.shape[0])]
        test_auc = roc_auc_score(labels, pred_scores)
        res.append([name, normal_class, test_auc])
        print('algorithm:{} \t normal class:{} \t auc:{:.2f}%'.format(name, normal_class, 100*test_auc))
    return res


def tune_nondeep_mnist():

    anomaly_ratio = 0.1
    res = []
    for normal_class in range(0, 10):
        data = mnist_data(normal_class, anomaly_ratio=anomaly_ratio)
        auc = non_deep_mnist(data, normal_class=normal_class)
        res.extend(auc)
    pd_res = pd.DataFrame(res, columns=['algorithm', 'normal_class', 'auc'])
    pd_res.to_csv('nondeep_mnist.csv')


def pd_plot(data_file, output_file):
    """
    plot the distribution of the output function values for both positive and negative samples
    :param data_file:
    :param out_put_png_name:
    :return:
    """
    data = pd.read_excel(data_file)
    # filter the abnormal values to make the figure clear to see
    lower_0 = data[data['labels']==0]['scores'].quantile(0.03)
    upper_0 = data[data['labels']==0]['scores'].quantile(0.97)
    aim_data_0 = data[data['labels']==0]['scores']
    out_data_0 = aim_data_0[(aim_data_0 < upper_0) & (aim_data_0 > lower_0)]
    plt.hist(out_data_0, bins=50,normed=True, label='negative samples', log=True)
    lower_1 = data[data['labels']==1]['scores'].quantile(0.03)
    upper_1 = data[data['labels']==1]['scores'].quantile(0.97)
    aim_data_1 = data[data['labels']==1]['scores']
    out_data_1 = aim_data_1[(aim_data_1 < upper_1) & (aim_data_1 > lower_1)]
    plt.hist(out_data_1, bins=50,normed=True, label='positive samples', log=True)
    plt.xlabel('log_px')
    plt.legend()
    plt.ylabel('probability density')
    plt.title('auc: 63%')
    plt.savefig(output_file)


if __name__ == '__main__':
    # data_name = 'texture'
    # normal_class = (2, ) if data_name == 'texture' else (1,)
    # data = get_uci_data(data_name, normal_class=normal_class)  # if 'texture', normal_class = (2,), otherwise = (1,)
    # non_deep_methods(data)
    # robust_realnvp(data, n_blocks=5, hidden_size=5)
    # search_parameters(data)
    # robust_svm(data)
    # print('Dataset: {}'.format(data_name))
    # tune_nondeep_mnist()
    tune_realnvp_mnist()




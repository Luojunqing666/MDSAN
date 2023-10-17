from torchvision import datasets, transforms
import torch
import os
import numpy as np
import pandas as pd
import os
import tqdm
from scipy.io import loadmat
from SequenceDatasets import dataset
from sequence_aug import *
from sklearn.utils import shuffle
signal_size=1024
dataname= {0:["97.mat","105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat","234.mat"],  # 1797rpm
           1:["98.mat","106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat","235.mat"],  # 1772rpm
           2:["99.mat","107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat","236.mat"],  # 1750rpm
           3:["100.mat","108.mat", "121.mat","133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat","237.mat"]}  # 1730rpm
datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]
axis = ["_DE_time", "_FE_time", "_BA_time"]
label = [i for i in range(0, 10)]
transfer_task=[[3], [2]]

def load_training(root_path, dir, batch_size, kwargs):
    if dir=="Source domain":
        list_data = get_filesz_train(root_path, transfer_task[0])
    else:
        list_data = get_filesz_train(root_path, transfer_task[1])
    data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
    source_train = dataset(list_data=data_pd)
    train_loader = torch.utils.data.DataLoader(source_train, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    # list_data = get_filesz1_yest(root_path, transfer_task[1])
    list_data = get_filesz1_yest(root_path, transfer_task[1])
    data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
    test_train = dataset(list_data=data_pd)
    test_loader = torch.utils.data.DataLoader(test_train, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader

def get_filesz_train(root, N):
    data = []
    lab =[]
    for k in range(len(N)):
        for n in range(len(dataname[N[k]])):
            if n==0:
               path1 =os.path.join(root,datasetname[3], dataname[N[k]][n])
            else:
                path1 = os.path.join(root,datasetname[0], dataname[N[k]][n])
            data1, lab1 = data_loadz(path1,dataname[N[k]][n],label=label[n])
            data += data1
            lab +=lab1
    #data = 2 * (data - min(map(min, data))) / (max(map(max, data)) - min(map(min, data))) + -1
    return [data, lab]
def data_loadz(filename, axisname, label):
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]
    fl = 2*(fl-fl.min())/(fl.max()-fl.min())+-1 #-1-1
    #fl =(fl - fl.min()) / (fl.max() - fl.min()) + -1  # -1-1
    #fl = (fl-fl.mean())/fl.std()
    data = []
    lab = []
    start, end = 0, signal_size
    # while end <= 102400:#121200
    while end <= fl.shape[0]-signal_size:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size
    return data, lab

def get_filesz1_yest(root, N):
    data = []
    lab =[]
    for k in range(len(N)):
        for n in range(len(dataname[N[k]])):
            if n==0:
               path1 =os.path.join(root,datasetname[3], dataname[N[k]][n])
            else:
                path1 = os.path.join(root,datasetname[0], dataname[N[k]][n])
            data1, lab1 = data_loadz1(path1,dataname[N[k]][n],label=label[n])
            data += data1
            lab +=lab1
    #data = 2 * (data - min(map(min, data))) / (max(map(max, data)) - min(map(min, data))) + -1
    return [data, lab]
def data_loadz1(filename, axisname, label):
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]
    fl = (fl-fl.min())/(fl.max()-fl.min())+-1 #-1-1
    fl = 2*(fl - fl.min()) / (fl.max() - fl.min()) + -1  # -1-1
    #fl = (fl-fl.mean())/fl.std()
    data = []
    lab = []
    start, end = 0, signal_size
    # while end <= 102400:#121200
    while end <= fl.shape[0]-signal_size:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size
    return data, lab
    # start, end = 102400, 102400+signal_size
    # for i in range(10):#121200
    #     data.append(fl[start:end])
    #     lab.append(label)
    #     start += signal_size
    #     end += signal_size
    # return data, lab
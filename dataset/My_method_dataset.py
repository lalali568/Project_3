import numpy as np
import pandas as pd
#from sympy.plotting.pygletplot.util import dot_product
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import MinMaxScaler


class My_method_dataset(Dataset):
    """
    构建dataset
    """
    def __init__(self, data,trend_data,label,label_timestamp):
        self.data=data
        self.trend_data=trend_data
        self.label=label
        self.label_timestamp=label_timestamp
        #self.cos_mat=cos_mat

    def __getitem__(self, index):

        return  torch.tensor(self.data[index],dtype=torch.float32),\
                torch.tensor(self.trend_data[index],dtype=torch.float32),\
                torch.tensor(self.label[index],dtype=torch.float32),\
                torch.tensor(self.label_timestamp[index],dtype=torch.float32)

    def __len__(self):
        return len(self.data)

def slideing_window(data,window_size,stride):
    """
    通过滑动窗口获取样本
    :param data:
    :param window_size:
    :param stride:
    :return:
    """
    windows_data=[]
    lenth=len(data)
    start=0
    end=window_size
    while end<lenth:
        windows_data.append(data[start:end])
        start+=stride
        end+=stride
    windows_data = np.array(windows_data)
    return windows_data
def process_labels(labels,window_size,stride):
    """
    这个函数是，提取每个窗口的label,如何有异常，则这个窗口的label为1
    :param labels:
    :param window_size:
    :param stride:
    :return:
    """
    window_labels=[]
    lenth = len(labels)
    start = 0
    end = window_size
    while end < lenth:
        if sum(labels[start:end]>0):
            window_labels.append(1)
        else:
            window_labels.append(0)
        start += stride
        end += stride
    window_labels = np.array(window_labels)
    return window_labels


def moving_average_with_padding(sequence, trend_size, padding_type='zero'):
    """
    移动平均实现，支持不同的边界填充方式
    :param sequence: 输入时间序列
    :param trend_size: 滑动窗口大小
    :param padding_type: 边界填充方式，'zero', 'reflect', or 'edge'
    :return: 平滑后的序列
    """
    if trend_size % 2 == 0:
        raise ValueError("窗口大小必须是奇数")

    for i in range(sequence.shape[1]):
        col=sequence[:,i]
        # 根据填充类型设置填充方式
        padding_size = trend_size // 2
        if padding_type == 'zero':
            padded_sequence = np.pad(col, (padding_size, padding_size), mode='constant', constant_values=0)
        elif padding_type == 'reflect':
            padded_sequence = np.pad(col, (padding_size, padding_size), mode='reflect')
        elif padding_type == 'edge':
            padded_sequence = np.pad(col, (padding_size, padding_size), mode='edge')
        else:
            raise ValueError("Unsupported padding type")

        # 滑动窗口取平均
        smooth_sequence = np.convolve(padded_sequence, np.ones(trend_size) / trend_size, mode='valid')
        sequence[:,i]=smooth_sequence
    return sequence

def pick_normal_data(train_data,train_trend_data,train_label):
    """
    这个函数把包含异常数据的样本挑选出来去除，用来测试在训练数据中，不让模型看到异常数据会不会结果变好
    :param train_data:
    :param train_trend_data:
    :param train_label:
    :return:
    """
    mask = np.array(list(map(int, np.any(train_label > 0, axis=1))))
    train_data = train_data[mask == 0]
    train_trend_data = train_trend_data[mask == 0]
    train_label = train_label[mask == 0]
    return train_data,train_trend_data,train_label

def temporal_cos_sim(config,data):
    """这个函数是把原本的时序转换为余弦相似度的"""
    slice_indices = [(j * config['padding_stride'], j * config['padding_stride'] + config['padding_size'])
                     for j in range((config['window_size'] - config['padding_size']) // config['padding_stride'])]

    # Preallocate the output matrix
    temporal_cos_sim_mat = np.zeros(
        (data.shape[0], len(slice_indices), config['input_size'], config['input_size']))

    for i in range(data.shape[0]):
        # Extract all slices at once for this train_data row
        matrices = [data[i, start:end, :] for start, end in slice_indices]
        matrices = np.array(matrices)  # Convert list of matrices to a single 3D array

        # Transpose the last two dimensions to prepare for dot product
        matrices_t = matrices.transpose(0, 2, 1)

        # Compute dot products for all slices at once
        dot_products = np.matmul(matrices_t, matrices)

        # Compute norms and avoid division by zero
        norms = np.linalg.norm(matrices_t, axis=2, keepdims=True)
        norms[norms == 0] = 1  # Replace 0 with 1 to prevent division by zero

        # Compute cosine similarity matrices for all slices at once
        cos_mats = dot_products / (np.matmul(norms, norms.transpose(0, 2, 1)))
        temporal_cos_sim_mat[i] = np.nan_to_num(cos_mats, nan=0)
    return temporal_cos_sim_mat

def get_dataset(config):

    # 加载数据，因为不知道数据的正常或者异常情况，所以用所有数据进行归一化，且使用MinMax归一化
    if config['dataset']=='SMD':
        data=np.loadtxt(config['test_data_path'],delimiter=',')
        scalar=MinMaxScaler()
        data=scalar.fit_transform(data)
    if config['dataset']=='SWAT':
        data=pd.read_csv(config['test_data_path'],low_memory=False)
        data.columns=data.columns.str.strip()
        data.drop(['Timestamp','Normal/Attack'],axis=1,inplace=True)
        scalar=MinMaxScaler()
        data=scalar.fit_transform(data)
    if config['dataset']=='WADI':
        data=pd.read_csv(config['test_data_path'],low_memory=False,skiprows=[0])
        data.columns=data.columns.str.strip()
        data.drop(['Row','Date','Time','Attack LABLE (1:No Attack, -1:Attack)'],axis=1,inplace=True)
        #所有数据的最后两行为空，删除最后两行，然后还有几列的数据完全为空，将其删除
        data=data[:-2]
        data.dropna(axis=1,inplace=True)
        scalar=MinMaxScaler()
        data=scalar.fit_transform(data)




    #得到trend
    trend_data=moving_average_with_padding(data, config['trend_size'], padding_type='edge')
    labels=np.loadtxt(config['test_label_path'],delimiter=',')


    #划分数据集
    #训练数据
    train_data=data[:int(len(data)*config['data_ratio'])]
    train_trend_data=trend_data[:int(len(data)*config['data_ratio'])]
    train_label=labels[:int(len(data)*config['data_ratio'])]
    train_data=slideing_window(train_data,config['window_size'],config['stride'])
    train_trend_data=slideing_window(train_trend_data,config['window_size'],config['stride'])
    train_label_timestamp = slideing_window(train_label, config['window_size'], config['stride'])
    train_label=process_labels(train_label,config['window_size'],config['stride'])
    #train_cos_mat=temporal_cos_sim(config,train_data)  #测试了一下，感觉效果一般,先不加了


    #测试数据
    test_data=data[int(len(data)*config['data_ratio']):]
    test_trend_data=trend_data[int(len(data)*config['data_ratio']):]
    test_label=labels[int(len(data)*config['data_ratio']):]
    test_data=slideing_window(test_data,config['window_size'],config['window_size'])
    test_trend_data=slideing_window(test_trend_data,config['window_size'],config['window_size'])
    test_label_timestamp=slideing_window(test_label,config['window_size'],config['window_size'])
    test_label=process_labels(test_label,config['window_size'],config['window_size'])
    #test_cos_mat=temporal_cos_sim(config,test_data) #测试了一下，感觉效果一般,先不加了




    train_data_dataset=My_method_dataset(train_data,train_trend_data,train_label,train_label_timestamp)
    test_data_dataset=My_method_dataset(test_data,test_trend_data,test_label,test_label_timestamp)
    return train_data_dataset,test_data_dataset






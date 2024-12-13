import numpy as np
from torch.utils.data import Dataset


class MTGFLOW_dataset(Dataset):
    """
    构建dataset
    """
    def __init__(self, data,label):
        self.data=data
        self.label=label

    def __getitem__(self, index):

        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)

def slideing_window(data,labels,window_size,stride):
    windows_data=[]
    windows_labels=[]
    lenth=len(data)
    start=0
    end=window_size
    while end<lenth:
        windows_data.append(data[start:end])
        windows_labels.append(labels[start:end])
        start+=stride
        end+=stride
    windows_data = np.array(windows_data)
    windows_labels = np.array(windows_labels)
    windows_labels=np.max(windows_labels,axis=1)
    return windows_data,windows_labels

def get_dataset(config):

    # 加载数据
    data=np.loadtxt(config['test_data_path'],delimiter=',')
    labels=np.loadtxt(config['test_label_path'],delimiter=',')

    #划分数据集
    train_data=data[:int(len(data)*0.6)]
    train_label=labels[:int(len(data)*0.6)]
    train_data,train_label=slideing_window(train_data,train_label,config['window_size'],config['stride'])

    test_data=data[int(len(data)*0.6):]
    test_label=labels[int(len(data)*0.6):]
    test_data,test_label=slideing_window(test_data,test_label,config['window_size'],config['stride'])


    train_data_dataset=MTGFLOW_dataset(train_data,train_label)
    test_data_dataset=MTGFLOW_dataset(test_data,test_label)
    return train_data_dataset,test_data_dataset,






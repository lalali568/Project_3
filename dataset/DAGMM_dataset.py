import numpy as np
from torch.utils.data import Dataset
class DAGMM_dataset(Dataset):
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


def get_dataset(config):
    """处理孤立森林中的数据需求"""
    # 加载数据
    test_data=np.loadtxt(config['test_data_path'],delimiter=',')
    labels=np.loadtxt(config['test_label_path'],delimiter=',')

    #窗口滑动
    windows_data=[]
    windows_labels=[]
    n=len(test_data)
    start=0
    end=config['window_size']
    while end<n:
        windows_data.append(test_data[start:end])
        windows_labels.append(max(labels[start:end]))
        start+=config['stride']
        end+=config['stride']
    windows_data = np.array(windows_data)
    windows_labels = np.array(windows_labels)
    #前80%作为测试集，后20%为验证集
    test_data=windows_data[:int(len(windows_data)*0.8)]
    test_label=windows_labels[:int(len(windows_data)*0.8)]
    val_data=windows_data[int(len(windows_data)*0.8):]
    val_label=windows_labels[int(len(windows_data)*0.8):]
    test_data_dataset=DAGMM_dataset(test_data,test_label)
    val_data_dataset=DAGMM_dataset(val_data,val_label)
    return test_data_dataset,test_data_dataset,val_data_dataset



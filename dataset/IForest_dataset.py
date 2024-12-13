import numpy as np


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

    return np.array(windows_data),np.array(windows_labels)



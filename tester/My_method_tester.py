import torch
from tqdm import tqdm
import numpy as np
from utils import plots
import matplotlib.pyplot as plt

def calculate_entropy_per_column(matrix, bins=20):
    """
    计算矩阵每一列的信息熵,希望的是信息熵越小的越好,越小表示这列能捕捉的异常越清晰

    参数:
    matrix: np.ndarray
        输入数据矩阵，形状为 (38, 200)
    bins: int
        分箱数量，默认 10

    返回:
    entropies: np.ndarray
        每一列的信息熵，形状为 (200,)
    """
    entropies = []
    for col in matrix.T:  # 转置矩阵，逐列处理
        # 统计每列的直方图分布
        histogram, _ = np.histogram(col, bins=bins, density=True)
        # 计算概率分布
        probabilities = histogram / np.sum(histogram)
        # 过滤掉概率为零的值
        probabilities = probabilities[probabilities > 0]
        # 计算信息熵
        entropy_value = -np.sum(probabilities * np.log2(probabilities))
        entropies.append(entropy_value)
    return np.array(entropies)


def tester(config,model,device,test_loader):
    with torch.no_grad():
        model.eval()
        labels=[]
        recon_data=[]
        orig_data=[]
        anomaly_score=[]#每个样本的损失值
        recon_losses=[]
        time_stamp_labels=[]
        for data, trend_data, test_label,test_timestamp_label in test_loader:
            data, trend_data, test_label=data.to(device), trend_data.to(device), test_label.to(device)
            trend,detail=model(data)
            output=trend+detail

            recon_error=(((output-data)**2).sum(dim=1))
            anomaly_score.append(recon_error.cpu().numpy())


            # 把（128，30，38）转换为(128*30,38)
            output=output.contiguous().view(-1,output.shape[-1])
            data=data.contiguous().view(-1,data.shape[-1])
            recon_loss = (output - data) ** 2
            recon_losses.append(recon_loss.cpu().numpy())

            recon_data.append(output.cpu().numpy())
            orig_data.append(data.cpu().numpy())
            labels.append(test_label.contiguous().view(-1,1).cpu().numpy())

            time_stamp_labels.append(test_timestamp_label.contiguous().view(-1,1).cpu().numpy())



        recon_losses=np.concatenate(recon_losses).squeeze()
        entropy = calculate_entropy_per_column(recon_losses)

        #根据entropy选择有明显区分度的特征
        select_var=np.where(entropy<0.05)
        anomaly_score = np.concatenate(anomaly_score).squeeze()
        anomaly_score=anomaly_score[:,select_var].squeeze()
        anomaly_score=anomaly_score.sum(axis=1)

        labels=np.concatenate(labels).squeeze()
        recon_data=np.concatenate(recon_data).squeeze()
        orig_data=np.concatenate(orig_data).squeeze()
        time_stamp_labels=np.concatenate(time_stamp_labels).squeeze()
    #重构数据，原始数据，重构损失可视化
    if config['plot_flag']:
        plots.plot_pre_result(config,recon_data,orig_data,recon_losses,time_stamp_labels)
        plots.plot_total_loss(config,anomaly_score,labels)

    return anomaly_score,labels
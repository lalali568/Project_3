import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def tester(config,model,device,test_loader):
    with torch.no_grad():
        model.eval()
        anomaly_score=[]
        labels=[]
        for batch_data, test_label in tqdm(test_loader):
            batch_data=torch.as_tensor(batch_data, dtype=torch.float32)
            batch_data=batch_data.transpose(1,2)
            batch_data=batch_data.unsqueeze(-1)
            batch_data = batch_data.to(device)
            labels.append(test_label.numpy())
            loss=-model.test(batch_data.to(device))
            anomaly_score.append(loss.detach().cpu().numpy())
        anomaly_score=np.concatenate(anomaly_score).squeeze()
        labels=np.concatenate(labels)
    return anomaly_score,labels
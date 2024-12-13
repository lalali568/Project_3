import torch
import torch.nn as nn
from spyder_kernels.utils.lazymodules import numpy
from tqdm import tqdm
import numpy as np



def tester(config,model,device,test_loader):

    with torch.no_grad():
        model.eval()
        anomaly_score=[]
        labels=[]
        for batch_data, test_label in tqdm(test_loader):
            batch_data=torch.tensor(batch_data, dtype=torch.float32)
            batch_data = batch_data.to(device)
            labels.append(test_label.numpy())
            _,reconstruction=model(batch_data.to(device))
            reconstruction_loss=nn.MSELoss(reduction='none')(batch_data,reconstruction)
            energy=torch.sum(reconstruction_loss,dim=1)
            energy=torch.sum(energy,dim=1)
            anomaly_score.append(energy.detach().cpu().numpy())
        anomaly_score=np.concatenate(anomaly_score).squeeze()
        labels=np.concatenate(labels)
    return anomaly_score,labels


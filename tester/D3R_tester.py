import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def _process_one_batch(model,batch_data, batch_time, batch_stable, device,p,train):
    criterion = nn.MSELoss(reduction='mean')
    batch_data = batch_data.float().to(device)
    batch_time = batch_time.float().to(device)
    batch_stable = batch_stable.float().to(device)

    if train:
        stable, _, recon = model(batch_data, batch_time, p)
        loss = 0.5 * criterion(stable, batch_stable) + \
               0.5 * criterion(recon, batch_data)
        return loss
    else:
        stable, trend, recon = model(batch_data, batch_time, 0.00)
        return stable, trend, recon



def tester(config,model,device,test_loader,init_loader):

    with torch.no_grad():
        model.eval()
        init_src, init_rec = [], []
        for (batch_data, batch_time, batch_stable, batch_label) in tqdm(init_loader):
            _, _, recon = _process_one_batch(model,batch_data, batch_time, batch_stable, device,config['p'], train=False)
            init_src.append(batch_data.detach().cpu().numpy()[:, -1, :])
            init_rec.append(recon.detach().cpu().numpy()[:, -1, :])

        test_label, test_src, test_rec = [], [], []
        for (batch_data, batch_time, batch_stable, batch_label) in tqdm(test_loader):
            _, _, recon = _process_one_batch(model,batch_data, batch_time, batch_stable, device,config['p'], train=False)
            test_label.append(batch_label.detach().cpu().numpy()[:, -1, :])
            test_src.append(batch_data.detach().cpu().numpy()[:, -1, :])
            test_rec.append(recon.detach().cpu().numpy()[:, -1, :])

    init_src = np.concatenate(init_src, axis=0)
    init_rec = np.concatenate(init_rec, axis=0)
    init_mse = (init_src - init_rec) ** 2

    test_label = np.concatenate(test_label, axis=0)
    test_src = np.concatenate(test_src, axis=0)
    test_rec = np.concatenate(test_rec, axis=0)
    test_mse = (test_src - test_rec) ** 2

    init_score = np.mean(init_mse, axis=-1, keepdims=True)
    test_score = np.mean(test_mse, axis=-1, keepdims=True)
    return init_score, test_score,test_label, test_src, test_rec


import torch
from torch import nn
from time import time
from tqdm import tqdm
from utils import earlystop
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

def trainer(config,device,optimizer,model,train_loader,valid_loader):
    path=f"checkpoints/{config[model]}_{config['dataset']}.pkl"
    early_stopping=earlystop.EarlyStop(path,config['patience'])

    for e in range(config.epochs):
        start = time()
        model.train()
        train_loss = []
        for (batch_data, batch_time, batch_stable, _) in tqdm(train_loader):
            optimizer.zero_grad()
            loss = _process_one_batch(model,batch_data, batch_time, batch_stable, device,config['p'], train=True)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            valid_loss = []
            for (batch_data, batch_time, batch_stable, _) in tqdm(valid_loader):
                loss = _process_one_batch(model,batch_data, batch_time, batch_stable, device,config['p'], train=True)
                valid_loss.append(loss.item())

        train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)
        end = time()
        print(f'Epoch: {e} || Train Loss: {train_loss:.6f} Valid Loss: {valid_loss:.6f} || Cost: {end - start:.4f}')

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            break

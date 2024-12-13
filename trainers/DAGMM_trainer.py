import torch
from time import time
from tqdm import tqdm
from utils import earlystop
import numpy as np


# DAGMM 损失函数
def dagmm_loss(x, x_reconstructed, energy):
    # 重构误差
    reconstruction_loss = torch.mean((x - x_reconstructed) ** 2)
    # 能量正则化
    energy_loss = torch.mean(energy)
    # 总损失
    total_loss = reconstruction_loss + 0.1 * energy_loss
    return total_loss


def trainer(config, device, optimizer, model, train_loader, valid_loader):
    save_path = f"checkpoints/{config['model']}_{config['dataset']}.pkl"
    early_stopping = earlystop.EarlyStop(save_path, config['patience'])

    for e in range(config['epochs']):
        total_loss = 0
        start = time()
        model.train()
        train_loss = []
        for (batch_data, _) in tqdm(train_loader):
            batch_data=torch.tensor(batch_data, dtype=torch.float32)
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            energy, reconstruction = model(batch_data)
            loss = dagmm_loss(batch_data, reconstruction, energy)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_loss.append(loss.item())
        print(f"Epoch [{e + 1}/{config['epochs']}], Loss: {total_loss / len(train_loader):.4f}")

        # 验证集部分
        with torch.no_grad():
            model.eval()
            valid_loss = []
            for (batch_data,_) in tqdm(valid_loader):
                batch_data=torch.tensor(batch_data, dtype=torch.float32)
                batch_data = batch_data.to(device)
                energy, reconstruction = model(batch_data)
                loss = dagmm_loss(batch_data, reconstruction, energy)
                valid_loss.append(loss.item())

        train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)
        end = time()
        print(f'Epoch: {e} || Train Loss: {train_loss:.6f} Valid Loss: {valid_loss:.6f} || Cost: {end - start:.4f}')

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            break
import torch
from time import time
from tqdm import tqdm
from utils import earlystop
from torch.nn.utils import clip_grad_value_
import numpy as np
import os

def trainer(config, device, optimizer, model, train_loader):
    save_path = f"checkpoints/{config['model']}_{config['dataset']}.pkl"

    for e in tqdm(range(config['epochs'])):
        total_loss = 0
        model.train()
        train_loss = []
        for (batch_data, label) in train_loader:
            batch_data=torch.as_tensor(batch_data, dtype=torch.float32)
            batch_data=batch_data.transpose(1,2)
            batch_data=batch_data.unsqueeze(-1)
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            loss = -model(batch_data)

            loss.backward()
            clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            total_loss += loss.item()
            train_loss.append(loss.item())
        print(f"Epoch [{e + 1}/{config['epochs']}], Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), save_path)



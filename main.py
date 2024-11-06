import os
import torch
from utils.evaluate import evaluate
from utils import Seed
from dataset import D3R_dataset
from models import D3R
from trainers import D3R_trainer
from tester import D3R_tester
from torch.utils.data import DataLoader
import pandas as pd
import yaml

"""#通过yaml文件读取参数, 并设置随机种子"""
with open("./config/D3R.yaml", 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    dataset_config=config.get(config['dataset'],{})
    config.update(dataset_config)
Seed.setSeed(config['seed'])
if config['device']=='cuda':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(config)
"""加载数据集"""
if config['model']=='D3R':
    train_dataset,valid_dataset,test_dataset,init_dataset=D3R_dataset.get_dataset(config)#读取dataset
    train_loader=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True,drop_last=False)
    valid_loader=DataLoader(valid_dataset,batch_size=config['batch_size'],shuffle=False,drop_last=False)
    test_loader=DataLoader(test_dataset,batch_size=config['batch_size'],shuffle=False,drop_last=False)
    init_loader=DataLoader(init_dataset,batch_size=config['batch_size'],shuffle=False,drop_last=False)

"""构建模型"""
if config['model']=='D3R':
    model=D3R.DDDR(config,device)
    model.to(device)
"""开始训练"""
if config['model']=='D3R':
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],weight_decay=1e-4)
    D3R_trainer.trainer(config,device,optimizer,model,train_loader,valid_loader)
"""开始测试"""
if config['model']=='D3R':
    # 加载模型
    model = D3R.DDDR(config, device)
    model.load_state_dict(torch.load(config['model_path']))
    init_score, test_score, test_label, test_src, test_rec=D3R_tester.tester(config, model,device , test_loader, init_loader)

"""计算结果"""
if config['model']=='D3R':
    res = evaluate(init_score.reshape(-1), test_score.reshape(-1), test_label.reshape(-1), q=config['q'])
    res['seed']=config['seed']
    res['test_dataset']=config['test_date_path']
    res_df=pd.DataFrame([res])
    csv_path=f"result/{config['model']}_{config['dataset']}.csv"
    if os.path.exists(csv_path):
        res_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        res_df.to_csv(csv_path, mode='w', header=True, index=False)




    

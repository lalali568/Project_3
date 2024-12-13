import os
import pandas as pd
import numpy as np
import yaml
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from numpy.ma.core import anomalies

from utils.evaluate import evaluate
from utils import Seed
from dataset import D3R_dataset,IForest_dataset,DAGMM_dataset,MTGFLOW_dataset,My_method_dataset
from sklearn.ensemble import IsolationForest

from models import D3R,DAGMM,MTGFLOW,My_method, My_method2
from trainers import D3R_trainer,DAGMM_trainer,MTGFLOW_trainer,My_method_trainer
from tester import D3R_tester,DAGMM_tester,MTGFLOW_tester,My_method_tester

from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,precision_recall_curve,auc
from datetime import datetime

"""#通过yaml文件读取参数, 并设置随机种子"""
with open('./config/My_method.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    dataset_config=config.get(config['dataset'],{})
    config.update(dataset_config)
Seed.setSeed(config['seed'])
if 'device' in config:
    if config['device']=='cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")



"""--------------------加载数据集-------------------------------"""
if config['model']=='D3R':
    train_dataset,valid_dataset,test_dataset,init_dataset=D3R_dataset.get_dataset(config)#读取dataset
    train_loader=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True,drop_last=False)
    valid_loader=DataLoader(valid_dataset,batch_size=config['batch_size'],shuffle=False,drop_last=False)
    test_loader=DataLoader(test_dataset,batch_size=config['batch_size'],shuffle=False,drop_last=False)
    init_loader=DataLoader(init_dataset,batch_size=config['batch_size'],shuffle=False,drop_last=False)
if config['model']=='IForest':
    windows_data,windows_labels=IForest_dataset.get_dataset(config)
    windows_data = windows_data.reshape(windows_data.shape[0], -1)  # 将窗口数据展平成二维数组
if config['model']=='DAGMM':
    train_loader,test_loader,valid_loader=DAGMM_dataset.get_dataset(config)
    train_loader=DataLoader(train_loader,batch_size=config['batch_size'],shuffle=True,drop_last=False)
    test_loader=DataLoader(test_loader,batch_size=config['batch_size'],shuffle=False,drop_last=False)
    valid_loader=DataLoader(valid_loader,batch_size=config['batch_size'],shuffle=False,drop_last=False)
if config['model']=='MTGFLOW':
    train_set,test_set=MTGFLOW_dataset.get_dataset(config)
    train_loader=DataLoader(train_set,batch_size=config['batch_size'],shuffle=True,drop_last=False)
    test_loader=DataLoader(test_set,batch_size=config['batch_size'],shuffle=False,drop_last=False)
if config['model']=='My_method':
    train_set,test_set=My_method_dataset.get_dataset(config)
    train_loader=DataLoader(train_set,batch_size=config['batch_size'],shuffle=True,drop_last=True)
    test_loader=DataLoader(test_set,batch_size=config['batch_size'],shuffle=False,drop_last=False)


"""-----------------------构建模型-----------------------------"""
if config['model']=='D3R':
    model=D3R.DDDR(config,device)
    model.to(device)
if config['model']=='IForest':
    model=IsolationForest(n_estimators=config['n_estimators'],max_samples=config['max_samples'],contamination=config['contamination'],random_state=config['seed'])
if config['model']=='DAGMM':
    model=DAGMM.DAGMM_model(config)
    model.to(device)
if config['model']=='MTGFLOW':
    model=MTGFLOW.MTGFLOW(config)
    model.to(device)
if config['model']=='My_method':
    model=My_method.My_model(config)
    model.to(device)


"""--------------------开始训练----------------------------------"""
if config['model']=='D3R':
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],weight_decay=1e-4)
    D3R_trainer.trainer(config,device,optimizer,model,train_loader,valid_loader)
if config['model']=='IForest':
    model.fit(windows_data)
if config['model']=='DAGMM':
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    DAGMM_trainer.trainer(config,device,optimizer,model,train_loader,valid_loader)
if config['model']=='MTGFLOW':
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
    MTGFLOW_trainer.trainer(config, device, optimizer, model, train_loader)
if config['model']=='My_method':
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    My_method_trainer.trainer(config, device, optimizer, model, train_loader)
"""----------------------开始测试----------------------------------"""
if config['model']=='D3R':
    # 加载模型
    model = D3R.DDDR(config, device)
    model.load_state_dict(torch.load(f"checkpoints/{config['model']}_{config['dataset']}.pkl"))
    model.to(device)
    init_score, test_score, test_label, test_src, test_rec=D3R_tester.tester(config, model,device , test_loader, init_loader)
if config['model']=='IForest':
    anomaly_score = model.decision_function(windows_data)
    anomalies=model.predict(windows_data)
    #把anomalies中1变成0，-1变成1
    anomalies[anomalies==1]=0
    anomalies[anomalies==-1]=1
if config['model']=='DAGMM':
    # 加载模型
    model=DAGMM.DAGMM_model(config)
    model.load_state_dict(torch.load(f"checkpoints/{config['model']}_{config['dataset']}.pkl"))
    model.to(device)
    anomaly_score,test_label = DAGMM_tester.tester(config, model, device, test_loader)
if config['model']=='MTGFLOW':
    model=MTGFLOW.MTGFLOW(config)
    state_dict = torch.load(f"checkpoints/{config['model']}_{config['dataset']}.pkl", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    anomaly_score,test_label = MTGFLOW_tester.tester(config, model, device, test_loader)
if config['model']=='My_method':
    model=My_method.My_model(config)
    state_dict = torch.load(f"checkpoints/{config['model']}_{config['dataset']}.pkl", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    anomaly_score,test_label = My_method_tester.tester(config, model, device, test_loader)

"""-----------------------计算结果----------------------------------"""
if config['model']=='D3R':
    res = evaluate(init_score.reshape(-1), test_score.reshape(-1), test_label.reshape(-1), q=config['q'])
    res['seed']=config['seed']
    res['test_dataset']=config['test_date_path']
    res_df=pd.DataFrame([res])
    csv_path=f"results/{config['model']}_{config['dataset']}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    res_df.to_csv(csv_path, mode='a', header=True, index=False)
if config['model']=='IForest':
    """记录precision,recall,f1,auc,prauc,time,seed"""
    result={}
    result['precision'], result['recall'], result['f1'] = precision_score(windows_labels, anomalies), recall_score(windows_labels, anomalies), f1_score(windows_labels, anomalies)
    result['auc'] = roc_auc_score(windows_labels, -anomaly_score)
    p,r,_ = precision_recall_curve(windows_labels, -anomaly_score)
    result['pr_auc']=auc(r, p)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result['time'] = current_time
    result['seed']=config['seed']
    result['test_dataset']=config['test_data_path']
    res_df=pd.DataFrame([result])
    csv_path=f"results/{config['model']}_{config['dataset']}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    res_df.to_csv(csv_path, mode='a', header=True, index=False)
    print(result)
if config['model']=='DAGMM':
    anomaly_threshold = np.percentile(anomaly_score, config['anomaly_ratio'])  # 设定阈值
    anomalies = (anomaly_score > anomaly_threshold).astype(int)
    #缩减一个维度
    """记录precision,recall,f1,auc,prauc,time,seed"""
    result = {}
    result['precision'], result['recall'], result['f1'] = precision_score(test_label, anomalies), recall_score(
        test_label, anomalies), f1_score(test_label, anomalies)
    result['auc'] = roc_auc_score(test_label, anomaly_score)
    p, r, _ = precision_recall_curve(test_label, anomaly_score)
    result['pr_auc'] = auc(r, p)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result['time'] = current_time
    result['seed'] = config['seed']
    result['test_dataset'] = config['test_data_path']
    res_df = pd.DataFrame([result])
    csv_path = f"results/{config['model']}_{config['dataset']}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    res_df.to_csv(csv_path, mode='a', header=True, index=False)
    print(result)
if config['model']=='MTGFLOW':
    """记录auc和pracuc"""
    result={}
    result['auc']=roc_auc_score(test_label,anomaly_score)
    p,r,_=precision_recall_curve(test_label,anomaly_score)
    result['pr_auc']=auc(r,p)
    current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result['time']=current_time
    result['seed']=config['seed']
    result['test_dataset']=config['test_data_path']
    res_df=pd.DataFrame([result])
    csv_path=f"results/{config['model']}_{config['dataset']}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not os.path.exists(csv_path):
        res_df.to_csv(csv_path,mode='a',header=True,index=False)
    else:
        res_df.to_csv(csv_path,mode='a',header=False,index=False)
    print(result)
if config['model']=='My_method':
    """记录auc和pracuc"""
    result = {}
    result['auc'] = roc_auc_score(test_label, anomaly_score)
    p, r, _ = precision_recall_curve(test_label, anomaly_score)
    result['pr_auc'] = auc(r, p)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result['time'] = current_time
    result['seed'] = config['seed']
    result['test_dataset'] = config['test_data_path']
    result['dynamic_pick']= config['dynamic_pick']
    res_df = pd.DataFrame([result])
    csv_path = f"results/{config['model']}_{config['dataset']}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not os.path.exists(csv_path):
        res_df.to_csv(csv_path, mode='a', header=True, index=False)
    else:
        res_df.to_csv(csv_path, mode='a', header=False, index=False)
    print(result)






    

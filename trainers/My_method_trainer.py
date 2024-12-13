import torch
from utils import earlystop,plots


def dynamic_loss_pick(config,trend, detail, data,trend_data, epoch):
    detail_data=data-trend_data
    trend_loss=(trend - trend_data) ** 2
    detail_loss=(detail_data - detail) ** 2
    recon_loss=trend_loss+detail_loss
    recon_loss_per_sample = recon_loss.sum(dim=-1).sum(dim=-1)

    # 动态调整阈值
    progress_ratio = epoch / config['epochs']
    dynamic_threshold = 1 - progress_ratio * 0.2

    recon_loss_per_sample_copy = recon_loss_per_sample.clone().detach()
    recon_loss_sorted, _ = torch.sort(recon_loss_per_sample_copy)
    top_threshold = recon_loss_sorted[int(len(recon_loss_sorted) * dynamic_threshold-1)]

    # 找到符合条件的索引
    indices = torch.nonzero(recon_loss_per_sample_copy < top_threshold, as_tuple=True)[0]
    shuffled_indices = indices[torch.randperm(indices.size(0))]

    return shuffled_indices


def trainer(config, device, optimizer, model, train_loader):
    save_path = f"checkpoints/{config['model']}_{config['dataset']}.pkl"
    train_loss = []

    minest_loss=float('inf')
    cur_patient=0
    for e in range(config['epochs']):
        total_loss = 0
        model.train()
        for data,trend_data, label, label_timestamp in train_loader:
            data, trend_data = data.to(device), trend_data.to(device)
            #首先选出容易训练的样本
            if config['dynamic_pick']==1:
                trend,detail = model(data)
                picked_indice=dynamic_loss_pick(config,trend,detail,data,trend_data,e)
                data,trend_data=data[picked_indice],trend_data[picked_indice]

            #正式训练
            detail_data=data-trend_data
            optimizer.zero_grad()
            trend,detail = model(data)
            trend_loss= ((trend-trend_data)**2).mean()
            detail_loss=((detail-detail_data)**2).mean()
            loss=trend_loss+detail_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        mean_loss=total_loss/len(train_loader)
        print(f"Epoch [{e + 1}/{config['epochs']}], Loss: {mean_loss}")
        # 早停，因为数据中包含异常值，感觉分验证集的意义不大，就以训练损失的平稳性来衡量吧
        if minest_loss == float('inf'):
            minest_loss = mean_loss
            torch.save(model.state_dict(), save_path)
        else:
            delta = minest_loss * 0.01  # 每一个epoch必须要有至少百分之1的变化
            if mean_loss < minest_loss - delta:
                torch.save(model.state_dict(), save_path)
                minest_loss = mean_loss
                cur_patient = 0
            else:
                cur_patient += 1
        if cur_patient >= config['patience']:
            print('Early Stop')
            break
        train_loss.append(total_loss / len(train_loader))
    if config['plot_flag']:
        plots.plot_loss(config,train_loss)





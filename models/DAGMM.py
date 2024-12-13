import torch
import torch.nn as nn



class DAGMM_model(nn.Module):
    def __init__(self,config):
        super(DAGMM_model, self).__init__()
        # 自编码器部分
        encoder_modules=[]
        for i in range(1,len(config['endcoder_hiddens'])):
            encoder_modules.append(nn.Linear(config['endcoder_hiddens'][i-1], config['endcoder_hiddens'][i]))
            encoder_modules.append(nn.ReLU())
        self.encoder_modules=nn.Sequential(*encoder_modules)
        # 解码
        decoder_modules=[]
        for i in range(1,len(config['decoder_hiddens'])):
            decoder_modules.append(nn.Linear(config['decoder_hiddens'][i-1], config['decoder_hiddens'][i]))
            decoder_modules.append(nn.ReLU())
        self.decoder_modules=nn.Sequential(*decoder_modules)
        # 估计网络
        estimation_modules=[]
        estimation_modules.append(nn.Linear(config['est_hiddens'][0], config['est_hiddens'][1]))
        estimation_modules.append(nn.ReLU())
        self.estimation_modules=nn.Sequential(*estimation_modules)


    def forward(self, x):
        # 编码
        latent = self.encoder_modules(x)
        # 重构
        x_reconstructed = self.decoder_modules(latent)
        # 计算重构误差
        reconstruction_error = torch.mean((x - x_reconstructed) ** 2, dim=2, keepdim=True)
        # 组合特征
        combined = torch.cat([latent, reconstruction_error], dim=2)
        # 估计网络
        energy = self.estimation_modules(combined)
        return energy, reconstruction_error
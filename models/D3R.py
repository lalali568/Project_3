import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
#%%
"""--------------------- 各种零件部件----------------------"""
"""embeddings"""
class PositionEmbedding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionEmbedding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.pe[:, : x.size(1), :]

        return self.norm(x)

class TimeEmbedding(nn.Module):
    def __init__(self, model_dim, time_num):
        super(TimeEmbedding, self).__init__()
        self.conv = nn.Conv1d(in_channels=time_num, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        return self.norm(x)

class DataEmbedding(nn.Module):
    def __init__(self, model_dim, feature_num):
        super(DataEmbedding, self).__init__()
        self.conv = nn.Conv1d(in_channels=feature_num, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x

"""attentions"""
class OrdAttention(nn.Module):
    def __init__(self, model_dim, atten_dim, head_num, dropout, residual):
        super(OrdAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual

        self.W_Q = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)

        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, Q, K, V):
        residual = Q.clone()

        Q = self.W_Q(Q).view(Q.size(0), Q.size(1), self.head_num, self.atten_dim)
        K = self.W_K(K).view(K.size(0), K.size(1), self.head_num, self.atten_dim)
        V = self.W_V(V).view(V.size(0), V.size(1), self.head_num, self.atten_dim)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.atten_dim))
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2)
        context = context.reshape(residual.size(0), residual.size(1), -1)
        output = self.dropout(self.fc(context))

        if self.residual:
            return self.norm(output + residual)
        else:
            return self.norm(output)

class MixAttention(nn.Module):
    def __init__(self, model_dim, atten_dim, head_num, dropout, residual):
        super(MixAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual

        self.W_Q_data = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_Q_time = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K_data = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K_time = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V_time = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)

        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, Q_data, Q_time, K_data, K_time, V_time):
        residual = Q_data.clone()

        Q_data = self.W_Q_data(Q_data).view(Q_data.size(0), Q_data.size(1), self.head_num, self.atten_dim)
        Q_time = self.W_Q_time(Q_time).view(Q_time.size(0), Q_time.size(1), self.head_num, self.atten_dim)
        K_data = self.W_K_data(K_data).view(K_data.size(0), K_data.size(1), self.head_num, self.atten_dim)
        K_time = self.W_K_time(K_time).view(K_time.size(0), K_time.size(1), self.head_num, self.atten_dim)
        V_time = self.W_V_time(V_time).view(V_time.size(0), V_time.size(1), self.head_num, self.atten_dim)

        Q_data, Q_time = Q_data.transpose(1, 2), Q_time.transpose(1, 2)
        K_data, K_time = K_data.transpose(1, 2), K_time.transpose(1, 2)
        V_time = V_time.transpose(1, 2)

        scores_data = torch.matmul(Q_data, K_data.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        scores_time = torch.matmul(Q_time, K_time.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        attn = nn.Softmax(dim=-1)(scores_data + scores_time)
        context = torch.matmul(attn, V_time)

        context = context.transpose(1, 2)
        context = context.reshape(residual.size(0), residual.size(1), -1)
        output = self.dropout(self.fc(context))

        if self.residual:
            return self.norm(output + residual)
        else:
            return self.norm(output)

"""transformer blocks"""
class TemporalTransformerBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(TemporalTransformerBlock, self).__init__()
        self.attention = OrdAttention(model_dim, atten_dim, head_num, dropout, True)

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.attention(x, x, x)

        residual = x.clone()
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        return self.norm(x + residual)

class SpatialTransformerBlock(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(SpatialTransformerBlock, self).__init__()
        self.attention = OrdAttention(window_size, atten_dim, head_num, dropout, True)

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.attention(x, x, x)
        x = x.permute(0, 2, 1)

        residual = x.clone()
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        return self.norm(x + residual)

class SpatialTemporalTransformerBlock(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(SpatialTemporalTransformerBlock, self).__init__()
        self.time_block = TemporalTransformerBlock(model_dim, ff_dim, atten_dim, head_num, dropout)
        self.feature_block = SpatialTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dropout)

        self.conv1 = nn.Conv1d(in_channels=2 * model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm1 = nn.LayerNorm(2 * model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        time_x = self.time_block(x)
        feature_x = self.feature_block(x)
        x = self.norm1(torch.cat([time_x, feature_x], dim=-1))

        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        return self.norm2(x)

"""decomposition block"""
class DecompositionBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, feature_num, head_num, dropout):
        super(DecompositionBlock, self).__init__()
        self.mixed_attention = MixAttention(model_dim, atten_dim, head_num, dropout, False)
        self.ordinary_attention = OrdAttention(model_dim, atten_dim, head_num, dropout, True)

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.fc1 = nn.Linear(model_dim, ff_dim, bias=True)
        self.fc2 = nn.Linear(ff_dim, feature_num, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, trend, time):
        stable = self.mixed_attention(trend, time, trend, time, time)
        stable = self.ordinary_attention(stable, stable, stable)

        residual = stable.clone()
        stable = self.activation(self.conv1(stable.permute(0, 2, 1)))
        stable = self.dropout(self.conv2(stable).permute(0, 2, 1))
        stable = self.norm1(stable + residual)

        trend = self.norm2(trend - stable)
        stable = self.fc2(self.activation(self.fc1(stable)))

        return stable, trend

"""data encoder和time encoder"""
class DataEncoder(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, block_num, head_num, dropout):
        super(DataEncoder, self).__init__()
        self.data_embedding = DataEmbedding(model_dim, feature_num)#主要是一维卷积的嵌入
        self.position_embedding = PositionEmbedding(model_dim)#sin cos的嵌入

        self.encoder_blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            self.encoder_blocks.append(
                SpatialTemporalTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dp)
            )

    def forward(self, x):
        x = self.data_embedding(x) + self.position_embedding(x)

        for block in self.encoder_blocks:
            x = block(x)

        return x

class TimeEncoder(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, time_num, block_num, head_num, dropout):
        super(TimeEncoder, self).__init__()
        self.time_embed = TimeEmbedding(model_dim, time_num)

        self.encoder_blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            self.encoder_blocks.append(
                TemporalTransformerBlock(model_dim, ff_dim, atten_dim, head_num, dp)
            )

    def forward(self, x):
        x = self.time_embed(x)

        for block in self.encoder_blocks:
            x = block(x)

        return x

class OffsetSubtraction(nn.Module):
    def __init__(self, window_size, feature_num, d):
        super(OffsetSubtraction, self).__init__()
        init_index = (torch.arange(window_size) + window_size).unsqueeze(-1).unsqueeze(-1)
        init_index = init_index.repeat(1, feature_num, 2 * d + 1)
        delay = torch.Tensor([0] + [i for i in range(1, d + 1)] + [-i for i in range(1, d + 1)]).int()
        delay = delay.unsqueeze(0).unsqueeze(0).repeat(window_size, feature_num, 1)
        self.index = init_index + delay
        self.d = d

    def forward(self, subed, sub):
        batch_size = subed.shape[0]
        index = self.index.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(sub.device)

        front = sub[:, 0:1, :].repeat(1, sub.shape[1], 1)
        end = sub[:, -1:, :].repeat(1, sub.shape[1], 1)
        sub = torch.cat([front, sub, end], dim=1)
        sub = torch.gather(sub.unsqueeze(-1).repeat(1, 1, 1, 2 * self.d + 1), dim=1, index=index)

        res = subed.unsqueeze(-1).repeat(1, 1, 1, 2 * self.d + 1) - sub
        res = torch.gather(res, dim=-1, index=torch.argmin(torch.abs(res), dim=-1).unsqueeze(-1))

        return res.reshape(subed.shape)
#%%
"""--------------------- DynamicDecomposition----------------------  """
class DynamicDecomposition(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, time_num, block_num, head_num, dropout,
                 d):
        super(DynamicDecomposition, self).__init__()
        self.data_encoder = DataEncoder(window_size, model_dim, ff_dim, atten_dim, feature_num, block_num,
                                        head_num, dropout)
        self.time_encoder = TimeEncoder(model_dim, ff_dim, atten_dim, time_num, block_num, head_num, dropout)

        self.decomposition_blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            self.decomposition_blocks.append(
                DecompositionBlock(model_dim, ff_dim, atten_dim, feature_num, head_num, dp)
            )

        self.minus = OffsetSubtraction(window_size, feature_num, d)

    def forward(self, data, time):
        residual = data.clone()

        data = self.data_encoder(data)
        time = self.time_encoder(time)
        stable = torch.zeros_like(residual).to(data.device)

        for block in self.decomposition_blocks:
            tmp_stable, data = block(data, time)#这个的data是一点一点迭代上去的
            stable = stable + tmp_stable#这个得到的temp也是一点一点加上去的

        trend = torch.mean(self.minus(residual, stable), dim=1).unsqueeze(1).repeat(1, data.shape[1], 1)#这里的self.minus的用法还要再多看看

        return stable, trend

#%%
""" --------------------Diffusion---------------------------"""
class Diffusion:
    def __init__(self, time_steps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.betas = torch.linspace(beta_start, beta_end, time_steps).float().to(device)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.one_minus_sqrt_alphas_cumprod = 1. - torch.sqrt(self.alphas_cumprod)

    @staticmethod
    def _extract(data, batch_t, shape):
        batch_size = batch_t.shape[0]
        out = torch.gather(data, -1, batch_t)

        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def q_sample(self, x_start, trend, batch_t, noise):
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, batch_t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, batch_t, x_start.shape)
        one_minus_sqrt_alphas_cumprod_t = self._extract(self.one_minus_sqrt_alphas_cumprod, batch_t, x_start.shape)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise + one_minus_sqrt_alphas_cumprod_t * trend

        return x_noisy
#%%
"""---------------------Reconstruction------------------------"""
class Reconstruction(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, time_num, block_num, head_num,
                 dropout):
        super(Reconstruction, self).__init__()
        self.time_embed = TimeEmbedding(model_dim, time_num)
        self.data_embedding = DataEmbedding(model_dim, feature_num)
        self.position_embedding = PositionEmbedding(model_dim)

        self.decoder_blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            self.decoder_blocks.append(
                SpatialTemporalTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dp)
            )

        self.fc1 = nn.Linear(model_dim, feature_num, bias=True)

    def forward(self, noise, trend, time):
        trend = self.data_embedding(trend)
        x = self.data_embedding(noise) - trend
        x = x + self.position_embedding(noise) + self.time_embed(time)

        for block in self.decoder_blocks:
            x = block(x)

        out = self.fc1(x + trend)

        return out


class DDDR(nn.Module):
    def __init__(self,config, device):
        super(DDDR, self).__init__()
        time_steps=config['time_steps']
        beta_start=config['beta_start']
        beta_end=config['beta_end']
        window_size = config['window_size']
        model_dim = config['model_dim']
        ff_dim = config['ff_dim']
        feature_num = config['feature_num']
        atten_dim = config['atten_dim']
        time_num = config['time_num']
        block_num = config['block_num']
        head_num = config['head_num']
        dropout = config['dropout']
        d=config['d']
        t=config['t']
        self.device = device
        self.window_size = window_size
        self.t = t

        self.dynamic_decomposition = DynamicDecomposition(
            window_size=window_size,
            model_dim=model_dim,
            ff_dim=ff_dim,
            atten_dim=atten_dim,
            feature_num=feature_num,
            time_num=time_num,
            block_num=block_num,
            head_num=head_num,
            dropout=dropout,
            d=d
        )

        self.diffusion = Diffusion(
            time_steps=time_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device
        )

        self.reconstruction = Reconstruction(
            window_size=window_size,
            model_dim=model_dim,
            ff_dim=ff_dim,
            atten_dim=atten_dim,
            feature_num=feature_num,
            time_num=time_num,
            block_num=block_num,
            head_num=head_num,
            dropout=dropout
        )

    def forward(self, data, time, p=0):
        disturb = torch.rand(data.shape[0], data.shape[2]) * p
        disturb = disturb.unsqueeze(1).repeat(1, self.window_size, 1).float().to(self.device)
        data = data + disturb

        stable, trend = self.dynamic_decomposition(data, time)

        bt = torch.full((data.shape[0],), self.t).to(self.device)
        sample_noise = torch.randn_like(data).float().to(self.device)
        noise_data = self.diffusion.q_sample(data, trend, bt, sample_noise)

        recon = self.reconstruction(noise_data, trend, time)

        return stable, trend - disturb, recon - disturb
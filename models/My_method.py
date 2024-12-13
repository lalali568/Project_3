import torch.nn as nn
import torch.nn.functional as F
import torch



class CNN_Patch_Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        # 不重叠的patch
        self.Q_CNN=nn.Conv1d(1,config['cnn_embedding_dim']*config['patch_attention_heads'],kernel_size=config['padding_size'],stride=config['padding_size'])
        self.K_CNN=nn.Conv1d(1,config['cnn_embedding_dim']*config['patch_attention_heads'],kernel_size=config['padding_size'],stride=config['padding_size'])
        self.scale=torch.sqrt(torch.tensor(config['embedding_dim'],dtype=torch.float32))

        self.V_Lin=nn.Linear(config['padding_size'],config['cnn_embedding_dim']*config['patch_attention_heads'])
        self.lin_out=nn.Linear(config['cnn_embedding_dim']*config['patch_attention_heads'],config['padding_size'])
        if config['embedding_dim']%config['padding_size']!=0:
            self.lin_output=nn.Linear(((config['embedding_dim']//config['padding_size'])+1)*config['padding_size'],config['embedding_dim'])
        else:
            self.lin_output = nn.Linear(
                (config['embedding_dim'] // config['padding_size']) * config['padding_size'],
                config['embedding_dim'])
    def forward(self,x):
        bs=x.shape[0]
        if self.config['embedding_dim'] % self.config['padding_size']!=0:#对patch进行处理，保证每个patch都能捕捉到
            pad_len = self.config['padding_size'] - (self.config['embedding_dim'] % self.config['padding_size'])
            pad_left=pad_len//2
            x=F.pad(x, (pad_left, pad_len - pad_left), mode='constant', value=0)

        x=x.reshape(-1,x.shape[-1]).unsqueeze(1)

        Q=self.Q_CNN(x).transpose(-1,-2)
        Q=Q.view(Q.shape[0],-1,self.config['patch_attention_heads'],self.config['cnn_embedding_dim']).transpose(1,2)
        K=self.K_CNN(x).transpose(-1,-2)
        K=K.view(K.shape[0],-1,self.config['patch_attention_heads'],self.config['cnn_embedding_dim']).transpose(1,2)


        att_map=F.softmax(torch.matmul(Q,K.transpose(-1,-2))/self.scale,dim=-1)
        x_v=x.view(x.shape[0],-1,self.config['padding_size'])

        V=self.V_Lin(x_v)
        V=V.view(x_v.shape[0],-1,self.config['patch_attention_heads'],self.config['cnn_embedding_dim']).transpose(1,2)
        x=torch.matmul(att_map,V)


        x=x.transpose(2,1)
        x=x.reshape(x.shape[0],x.shape[1],-1)
        x=self.lin_out(x).view(x.shape[0],-1)
        x=self.lin_output(x).view(bs,-1,self.config['embedding_dim'])
        return x


class Spatial_Temporal_Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.cnn_patch_attention=CNN_Patch_Attention(config)
        self.out_linear=nn.Linear(config['embedding_dim'],config['window_size'])
        self.Q=nn.Linear(config['embedding_dim'],config['embedding_dim']*config['spatial_temporal_heads'])
        self.K=nn.Linear(config['embedding_dim'],config['embedding_dim']*config['spatial_temporal_heads'])
        self.V=nn.Linear(config['embedding_dim'],config['embedding_dim']*config['spatial_temporal_heads'])
        self.d = self.config['embedding_dim']
        self.lin1=nn.Linear(config['embedding_dim']*config['spatial_temporal_heads'],config['embedding_dim'])


    def layer_norm(self, data):
        mean = data.mean(-1, keepdim=True)
        std = data.std(-1, keepdim=True)
        return (data - mean) / std

    def forward(self,x):
        #step1,空间上的注意力机制
        orig_x=x
        Q=self.Q(x)
        Q = Q.view(x.shape[0], -1, self.config['spatial_temporal_heads'], self.config['embedding_dim']).transpose(1,2)
        K=self.K(x)
        K = K.view(x.shape[0], -1, self.config['spatial_temporal_heads'], self.config['embedding_dim']).transpose(1,2)
        V=self.V(x)
        V = V.view(x.shape[0], -1, self.config['spatial_temporal_heads'], self.config['embedding_dim']).transpose(1,2)


        att=torch.matmul(Q,K.transpose(-1,-2))/self.d
        att=F.softmax(att,dim=-1)
        x=torch.matmul(att,V)


        x=x.transpose(2,1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x=self.lin1(x)
        x=self.layer_norm(x+orig_x)

        #step，时间上的注意力机制，使用cnn_patch_attentioh
        orig_x=x
        x=self.cnn_patch_attention(x)
        x=self.layer_norm(x+orig_x)

        return  x


class My_model(nn.Module):
    """这是主要的函数"""
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.embedding_lin=nn.Linear(config['window_size'],config['embedding_dim'])
        self.detail_spatial_temporal_transformer_layers=nn.ModuleList([Spatial_Temporal_Transformer(config) for _ in range(config['n_spatial_temporal_layers'])])
        self.trend_spatial_temporal_transformer_layers = nn.ModuleList([Spatial_Temporal_Transformer(config) for _ in range(config['n_spatial_temporal_layers'])])
        output_lin_1=[nn.Linear(config['embedding_dim'],config['window_size']),
                        nn.LeakyReLU(negative_slope=0.01),
                        nn.Linear(config['window_size'],config['window_size']),
                        nn.Sigmoid()]
        self.output_lin_1=nn.Sequential(*output_lin_1)
        output_lin_2 = [nn.Linear(config['embedding_dim'], config['window_size']),
                        nn.LeakyReLU(negative_slope=0.01),
                        nn.Linear(config['window_size'], config['window_size']),
                        nn.Sigmoid()]
        self.output_lin_2 = nn.Sequential(*output_lin_2)


    def forward(self, x):

        batch_size=x.shape[0]
        x=x.transpose(1,2)
        x=self.embedding_lin(x)
        for layer in self.detail_spatial_temporal_transformer_layers:
            detail = layer(x)
            detail=detail.view(batch_size,-1,detail.shape[-1])
        detail=self.output_lin_1(detail)
        detail=detail.transpose(1,2)

        for layer in self.trend_spatial_temporal_transformer_layers:
            trend=layer(x)
            trend=trend.view(batch_size,-1,trend.shape[-1])
        trend=self.output_lin_2(trend)
        trend = trend.transpose(1,2)

        return trend,detail



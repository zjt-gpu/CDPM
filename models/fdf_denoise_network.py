import math
import torch
from torch import nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaLayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb = SinusoidalPosEmb(emb_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(emb_dim, emb_dim*2)
        self.layernorm = nn.LayerNorm(emb_dim, elementwise_affine=False)
        
    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x

class Conv_MLP(nn.Module):
    def __init__(self, in_dim, emb_dim, dropout=0.1):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_dim, emb_dim, 3, stride=1, padding=1),
            nn.Dropout(p=dropout),
        )
    def forward(self, x):
        return self.sequential(x).transpose(1, 2)
    
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, emb_dim, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1, max_len, emb_dim)) 
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class LINEAR(nn.Module):
    def __init__(self, in_features, out_features, drop=0.1):
        super(LINEAR, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x): 
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 n_channel,
                 emb_dim=96,
                 dropout=0.1
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(emb_dim)
        
        self.LayerNorm = nn.LayerNorm(emb_dim)
        self.LINEAR = LINEAR(n_channel, n_channel)
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(emb_dim, emb_dim * 4)
        self.linear2 = nn.Linear(emb_dim * 4, emb_dim)

    def forward(self, x, timestep, mask=None, label_emb=None):
        
        x = self.ln1(x, timestep, label_emb)
        res1 = x.clone()
        x = self.LINEAR(x.permute(0, 2, 1)).permute(0, 2, 1) 
        res2 = x + res1
        x = self.activation(self.linear1(res2))
        x = self.dropout(x) 
        x = self.linear2(x)
        x = x + res2
        x = self.LayerNorm(x)
        
        return x 

class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        emb_dim=1024,
        n_layer=14,
        dropout=0.1
    ):
      super().__init__()
      self.n_feat = n_feat
      self.blocks = nn.Sequential(*[
            nn.Sequential(
                DecoderBlock(
                    n_channel=n_channel,
                    emb_dim=emb_dim,
                    dropout=dropout
                ),
                nn.LayerNorm(emb_dim) 
            ) for _ in range(n_layer)
        ])
      
    def forward(self, enc, t, padding_masks=None, label_emb=None):
        x = enc
        for block_idx in range(len(self.blocks)):
            x = self.blocks[block_idx][0](x, t, mask=padding_masks, label_emb=label_emb)
            x = self.blocks[block_idx][1](x)

        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    

class fdf_denoise_network(nn.Module):
    def __init__(
        self,
        n_feat,
        seq_len,
        pred_len,
        device,
        MLP_hidden_dim=256,
        emb_dim=1024,
        patch_len=4,
        n_layer=14,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        self.emb = Conv_MLP(n_feat, emb_dim, dropout=dropout)
        
        self.sparsity_threshold = 0.01
        
        self.seq_length = seq_len
        self.pred_length = pred_len
        self.patch_len = patch_len
        
        self.device = device

        self.pos_dec = LearnablePositionalEncoding(emb_dim, dropout=dropout, max_len=pred_len)
        self.decoder = Decoder(self.pred_length , n_feat, emb_dim, n_layer, dropout)
        self.feat_linear = nn.Linear(emb_dim, n_feat, bias = True)
        
        self.weight = nn.Parameter(torch.randn(1)) 
        
        self.MLP_hidden_size = MLP_hidden_dim
        
        self.mean_linear = nn.Sequential(
            nn.Linear(self.seq_length // self.patch_len, self.MLP_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.MLP_hidden_size, self.pred_length // self.patch_len)
        )
        self.var_linear = nn.Sequential(
            nn.Linear(self.seq_length // self.patch_len , self.MLP_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.MLP_hidden_size, self.pred_length // self.patch_len)
        )
    
    def forward(self, input, t, cond, padding_masks=None):
        
        batch_size, _, feature_dim = cond.shape
        
        num_patches = self.seq_length // self.patch_len
        cond_patches = cond[:, :num_patches * self.patch_len, :].view(batch_size, num_patches, self.patch_len, feature_dim)
        patch_mean = cond_patches.mean(dim=2)
        patch_var = cond_patches.var(dim=2, unbiased=False).sqrt()
        pred_mean = self.mean_linear(patch_mean.permute(0, 2, 1)).permute(0, 2, 1)
        pred_var = self.var_linear(patch_var.permute(0, 2, 1)).permute(0, 2, 1)
        pred_patches = self.pred_length // self.patch_len
        epsilon = torch.randn(batch_size, pred_patches, self.patch_len, feature_dim, device=cond.device)
        sampled = pred_mean.unsqueeze(2).repeat(1, 1, self.patch_len, 1) + epsilon * (pred_var).unsqueeze(2).repeat(1, 1, self.patch_len, 1)
        sampled = sampled.reshape(batch_size, self.patch_len * pred_patches, feature_dim)
        
        total_var = pred_var.sum(dim=1, keepdim=True)  
        var_ratio = (pred_var / total_var) * 0.5
        emb = self.emb(input)
        inp_dec = self.pos_dec(emb)
        output = self.decoder(inp_dec, t, padding_masks=padding_masks)
        output = self.feat_linear(output)
        
        result = self.weight * output + (1 - self.weight) * sampled
        
        return result

if __name__ == '__main__':
    pass

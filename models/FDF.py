import torch
import torch.nn as nn
from argparse import Namespace
from models.fdf_backbone import (
    Diffusion,
    series_decomposition,
    MultiLinearModel
)

class FDF(nn.Module):
    def __init__(self, args: Namespace):
        super(FDF, self).__init__()

        self.decom = series_decomposition(kernel_size = 5)
        self.input_len = args.input_len
        self.device = args.device
        self.pred_len = args.pred_len
        self.time_steps = args.time_steps

        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            feature_dim=args.feature_dim,
            seq_len=args.input_len,
            pred_len=args.pred_len,
            MLP_hidden_dim=args.MLP_hidden_dim,
            emb_dim=args.emb_dim,
            device=self.device,
            beta_scheduler=args.scheduler,
            patch_size=args.patch_size
        )
        self.eta = 0
        
        self.seq_len = args.input_len 
        self.trend_linear = MultiLinearModel(seq_len = args.input_len, pred_len = args.pred_len)
        
    def pred(self, x):
        batch_size, input_len, num_features = x.size()
        
        x_seq = x[:, :self.seq_len, :]
        x_means = x_seq.mean(1, keepdim=True).detach()
        x_enc = x_seq - x_means
        x_stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        
        x_norm = x - x_means
        x_norm /= x_stdev
        
        x_seq_input = x_norm[:, :self.seq_len, :]
        season_seq, trend_seq = self.decom(x_seq_input)
        x_pred = x_norm[:, -self.pred_len:, :]
        season_pred, trend_pred = self.decom(x_pred)
        
        trend_pred = self.trend_linear(trend_seq)
        
        # Noising Diffusion
        t = torch.randint(0, self.time_steps, (batch_size,), device=self.device) 
        noise_season = self.diffusion(season_pred, t) 
        season_pred = self.diffusion.pred(noise_season, t, season_seq)
        
        predict_x = trend_pred + season_pred
        
        dec_out = predict_x * \
                  (x_stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (x_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out
    

    def forecast(self, input_x):
        x = input_x[:, :self.seq_len, :]
        b, _, dim = x.shape
        shape = torch.zeros((b, self.pred_len, dim), dtype=torch.int, device=self.device)
        
        x_means = x.mean(1, keepdim=True).detach()
        x_enc = x - x_means
        x_stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= x_stdev
        season, trend = self.decom(x_enc)
        trend_pred_part = trend
        trend_pred = self.trend_linear(trend_pred_part)
        season_pred = self.diffusion.sample_infill(shape, self.time_steps, season)
        predict_x = trend_pred + season_pred
        dec_out = predict_x * \
                  (x_stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (x_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        x_pred = input_x[:, -self.pred_len:, :]
        season_pred_part, trend_pred_part = self.decom(x_pred)
        return dec_out

    def forward(self, x, task):
        if task == "train":
            return self.pred(x)  
        elif task == 'valid' or task == "test":
            return self.forecast(x)  
        else:
            raise ValueError(f"Invalid task: {task=}")

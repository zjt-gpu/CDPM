import math
import torch
import torch.nn as nn
from models.fdf_denoise_network import fdf_denoise_network
from functools import partial

class moving_avg(nn.Module):
    
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomposition(nn.Module):
    
    def __init__(self, kernel_size):
        super(series_decomposition, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Diffusion(nn.Module):
    def __init__(
        self,
        time_steps: int,
        feature_dim : int,
        seq_len : int,
        pred_len : int,
        MLP_hidden_dim : int,
        emb_dim : int,
        patch_size : int,
        device: torch.device,
        beta_scheduler: str = "cosine",
    ):
        super(Diffusion, self).__init__()
        self.device = device
        self.time_steps = time_steps
        self.seq_length = seq_len
        self.pred_length = pred_len

        if beta_scheduler == 'cosine':
            self.betas = self._cosine_beta_schedule().to(self.device)
        elif beta_scheduler == 'linear':
            self.betas = self._linear_beta_schedule().to(self.device)
        elif beta_scheduler == 'exponential':
            self.betas = self._exponential_beta_schedule().to(self.device)
        elif beta_scheduler == 'inverse_sqrt':
            self.betas = self._inverse_sqrt_beta_schedule().to(self.device)
        elif beta_scheduler == 'piecewise':
            self.betas = self._piecewise_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Unknown schedule type: {scheduler}")
        
        self.eta = 0
        self.alpha = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alpha, dim=0)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)
        
        self.denoise_net = fdf_denoise_network(feature_dim, seq_len, pred_len, device, MLP_hidden_dim, emb_dim, patch_size)
    
    def _cosine_beta_schedule(self, s=0.008):
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alpha_cumprod = (
            torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _linear_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, self.time_steps)

    def _exponential_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        steps = self.time_steps
        return beta_start * ((beta_end / beta_start) ** (torch.linspace(0, 1, steps)))

    def _inverse_sqrt_beta_schedule(self, beta_start=1e-4):
        steps = self.time_steps
        x = torch.arange(1, steps + 1)
        return torch.clip(beta_start / torch.sqrt(x), 0, 0.999)

    def _piecewise_beta_schedule(self, beta_values=[1e-4, 0.01, 0.02], segment_steps=[100, 200, 300]):
        assert len(beta_values) == len(segment_steps), "beta_values and segment_steps length mismatch"
        betas = [torch.full((steps,), beta) for beta, steps in zip(beta_values, segment_steps)]
        return torch.cat(betas)[:self.time_steps]
    
    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        t = t.to(a.device)  
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def noise(self, x, t):
        noise = torch.randn_like(x)
        gamma_t = self.gamma[t].unsqueeze(-1).unsqueeze(-1)  
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x, t):
        noisy_x, _ = self.noise(x, t)
        return noisy_x
    
    def pred(self, x, t, cond):
        if t == None:
            t = torch.randint(0, self.time_steps, (x.shape[0],), device=self.device)
        return self.denoise_net(x, t, cond)
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, t, cond, clip_x_start=False, padding_masks=None):

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.denoise_net(x, t, cond)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start
        
    @torch.no_grad()
    def sample_infill(self, shape, sampling_timesteps, cond, clip_denoised=True):
        batch_size, _, _ = shape.shape
        batch, device, total_timesteps, eta = shape[0], self.device, self.time_steps, self.eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  
        shape = shape
        denoise_series = torch.randn(shape.shape, device=device)

        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(denoise_series, time_cond, cond, clip_x_start=clip_denoised)

            if time_next < 0:
                denoise_series = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(denoise_series)

            denoise_series = pred_mean + sigma * noise

        return denoise_series

class MultiLinearModel(nn.Module):
    def __init__(self, seq_len, pred_len, num_loops=2):
        super(MultiLinearModel, self).__init__()

        self.linear_projection = nn.Linear(seq_len, pred_len, bias=True)
        self.weighted_linear = nn.Linear(num_loops, 1, bias=True)
        self.num_loops = num_loops

    def forward(self, input_data):
        transformed_data = [input_data.unsqueeze(-1)]
        
        for i in range(2, self.num_loops + 1):
            transformed = input_data.clone()
            transformed[:, 1, :] = torch.sign(input_data[:, 1, :]) * (torch.abs(input_data[:, 1, :]) ** (1 / i))
            transformed_data.append(transformed.unsqueeze(-1))
        
        concatenated_data = torch.cat(transformed_data, dim=-1)
        sequence_output = self.linear_projection(concatenated_data.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        output = self.weighted_linear(sequence_output).squeeze(-1)
        
        return output

from __future__ import annotations

import torch

from cofitok.configs import DiffusionConfig


class DiffusionSchedule:
    def __init__(self, config: DiffusionConfig, device: torch.device | str) -> None:
        self.config = config
        if config.schedule_type == "linear":
            betas = torch.linspace(
                config.beta_start,
                config.beta_end,
                config.num_train_timesteps,
                dtype=torch.float32,
                device=device,
            )
        elif config.schedule_type == "cosine":
            steps = config.num_train_timesteps + 1
            values = torch.linspace(0, config.num_train_timesteps, steps, device=device)
            cumulative = torch.cos(
                ((values / config.num_train_timesteps + 0.008) / 1.008) * torch.pi * 0.5
            ).pow(2)
            cumulative = cumulative / cumulative[0]
            betas = (1.0 - cumulative[1:] / cumulative[:-1]).clamp(1e-4, 0.999)
        else:
            raise ValueError(f"Unknown diffusion schedule_type: {config.schedule_type}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    @property
    def num_train_timesteps(self) -> int:
        return self.config.num_train_timesteps

    def sample_timesteps(
        self, batch_size: int, device: torch.device | str
    ) -> torch.Tensor:
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)

    def add_noise(
        self,
        clean_images: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sigma = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return alpha * clean_images + sigma * noise

    def predict_x0_from_epsilon(
        self,
        noisy_images: torch.Tensor,
        epsilon: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sigma = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return (noisy_images - sigma * epsilon) / alpha.clamp_min(1e-8)

    def snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_squared = self.alphas_cumprod[timesteps]
        sigma_squared = 1.0 - alpha_squared
        return alpha_squared / sigma_squared.clamp_min(1e-12)

    def min_snr_loss_weights(
        self,
        timesteps: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Return standard epsilon-prediction Min-SNR loss weights."""
        if gamma <= 0.0:
            return torch.ones_like(timesteps, dtype=torch.float32)
        snr = self.snr(timesteps).float()
        capped = torch.minimum(snr, snr.new_full(snr.shape, gamma))
        return capped / snr.clamp_min(1e-12)

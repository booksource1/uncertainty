import torch
import yaml
import sys
import os
from diffusers import AutoencoderKL

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from diffusion import create_diffusion
from isolated_nwm_infer import model_forward_wrapper
from models import CDiT_models

class NWMWrapper:
    def __init__(self, 
                 model_path, 
                 vae_path, 
                 device="cuda",
                 config_path=None,
                 diffusion_steps: int = 250):
        self.device = device
        
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "config/nwm_cdit_xl.yaml")
        
        # 确保 config_path 是绝对路径
        if not os.path.isabs(config_path):
            config_path = os.path.join(PROJECT_ROOT, config_path)
            
        print(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # 覆盖 load_path
        self.config["load_path"] = model_path
        
        # NWM 的 latent size 通常是 image size / 8 (对于 SD VAE)
        self.latent_size = self.config.get("image_size", 128) // 8
        self.context_size = self.config.get("context_size", 4)
        
        print(f"Loading model from {model_path}...")
        self.model = CDiT_models[self.config['model']](
            input_size=self.latent_size, 
            context_size=self.config['context_size']
        )
        
        ckp = torch.load(model_path, map_location='cpu', weights_only=False)
        if "ema" in ckp:
            print("Loading EMA weights...")
            self.model.load_state_dict(ckp["ema"], strict=True)
        elif "model" in ckp:
            print("Loading standard model weights...")
            # handle compiled model prefix
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckp['model'].items()}
            self.model.load_state_dict(state_dict, strict=True)
        else:
            print("Loading direct state dict...")
            self.model.load_state_dict(ckp, strict=True)
            
        self.model.to(device)
        self.model.eval()
        
        print(f"Loading VAE from {vae_path}...")
        self.vae = AutoencoderKL.from_pretrained(vae_path).to(device)
        self.vae.eval()
        
        # NOTE: create_diffusion expects timestep_respacing as a string like "250", "50", etc.
        self.diffusion_steps = int(diffusion_steps)
        self.diffusion = create_diffusion(timestep_respacing=str(self.diffusion_steps))
        
        # Pre-compute rel_t (assuming single step prediction or fixed time horizon)
        # Interactive test uses 1/128.
        self.rel_t_base = float(1.0 / 128.0)
        
    def predict(self, context_images, action, generator=None, init_noise=None):
        """
        Args:
            context_images: [B, 4, 3, H, W] tensor (normalized)
            action: [B, action_dim] tensor
            generator: torch.Generator
        Returns:
            pred_image: [B, 3, H, W] tensor (range [-1, 1] approximately)
        """
        if action.ndim == 2:
            action = action.unsqueeze(1) # [B, 1, dim]

        # 确保数据在正确的设备上
        context_images = context_images.to(self.device)
        action = action.to(self.device)

        with torch.no_grad():
            # rel_t should match batch size
            rel_t = (torch.ones(action.shape[0], device=self.device) * self.rel_t_base) * self.diffusion_steps
            outputs = model_forward_wrapper(
                (self.model, self.diffusion, self.vae),
                context_images,
                action,
                None, 
                self.latent_size,
                self.device,
                self.context_size,
                num_goals=1,
                rel_t=rel_t,
                progress=False,
                return_latent=False,
                generator=generator,
                init_noise=init_noise,
                x_cond_override=None,
                deterministic_vae=True
            )
        
        return outputs

    @torch.no_grad()
    def encode_x_cond(self, context_images: torch.Tensor) -> torch.Tensor:
        """
        仅做一次 VAE 编码，得到可复用的条件 latent（避免 multi-seed 重复编码，且默认使用 mean 以去除 VAE 采样噪声）。

        Args:
            context_images: [B, T, 3, H, W] normalized tensor
        Returns:
            x_cond: [B, num_cond, 4, latent, latent] (num_goals=1 情况下)
        """
        x = context_images.to(self.device)
        B, T = x.shape[:2]
        x = x.flatten(0, 1)
        posterior = self.vae.encode(x).latent_dist
        lat = posterior.mean.mul(0.18215).unflatten(0, (B, T))
        x_cond = lat[:, : self.context_size].unsqueeze(1).expand(B, 1, self.context_size, lat.shape[2], lat.shape[3], lat.shape[4]).flatten(0, 1)
        return x_cond

    def predict_from_x_cond(
        self,
        x_cond: torch.Tensor,
        action: torch.Tensor,
        init_noise: torch.Tensor,
        noise_sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        使用预先编码好的 x_cond 进行预测（不重复 VAE encode），支持 multi-seed 并行。

        Args:
            x_cond: [B, num_cond, 4, latent, latent]
            action: [B, 3] or [B, 1, 3]
            init_noise: [B, 4, latent, latent]
        """
        if action.ndim == 2:
            action = action.unsqueeze(1)
        action = action.to(self.device)
        rel_t = (torch.ones(action.shape[0], device=self.device) * self.rel_t_base) * self.diffusion_steps
        with torch.no_grad():
            outputs = model_forward_wrapper(
                (self.model, self.diffusion, self.vae),
                # curr_obs 仅用于占位（不会被使用），保持最小张量以减少传输
                torch.empty((1, 1, 3, 1, 1), device=self.device),
                action,
                None,
                self.latent_size,
                self.device,
                self.context_size,
                num_goals=1,
                rel_t=rel_t,
                progress=False,
                return_latent=False,
                generator=None,
                init_noise=init_noise,
                noise_sequence=noise_sequence,
                x_cond_override=x_cond,
                deterministic_vae=True,
            )
        return outputs

    @torch.no_grad()
    def sample_with_probes_from_x_cond(
        self,
        x_cond: torch.Tensor,
        action: torch.Tensor,
        init_noise: torch.Tensor,
        probe_steps: list[int],
        noise_sequence: torch.Tensor | None = None,
        stop_at_max_probe: bool = False,
        progress: bool = False,
    ):
        """
        Early-Exit/Probe sampling:
        - Runs diffusion sampling once (up to diffusion_steps).
        - At selected step indices (1-based), captures the model's pred_xstart (x0) in latent space.

        Args:
            x_cond: [B, num_cond, 4, latent, latent]
            action: [B, 3] or [B, 1, 3]
            init_noise: [B, 4, latent, latent] (z_0)
            probe_steps: list of 1-based step indices in the *spaced* diffusion (e.g., [1,5,10,25,50])
        Returns:
            final_latent: [B, 4, latent, latent]
            probes: dict[int, torch.Tensor] mapping step->pred_xstart latent [B,4,latent,latent]
        """
        if action.ndim == 2:
            action = action.unsqueeze(1)  # [B,1,3]
        action = action.to(self.device)
        x_cond = x_cond.to(self.device)
        init_noise = init_noise.to(self.device)
        rel_t = (torch.ones(action.shape[0], device=self.device) * self.rel_t_base) * self.diffusion_steps
        # y should be [N,3] for ActionEmbedder (avoid flatten corner-cases)
        y = action[:, 0, :]

        # sanitize probe steps
        probe_steps = sorted({int(s) for s in probe_steps if int(s) >= 1})
        probes = {}
        if len(probe_steps) == 0:
            # fall back to just final sample
            latent_final = self.diffusion.p_sample_loop(
                self.model,
                init_noise.shape,
                init_noise,
                clip_denoised=False,
                model_kwargs=dict(y=y, x_cond=x_cond, rel_t=rel_t),
                device=self.device,
                progress=progress,
            )
            return latent_final, probes

        max_probe = max(probe_steps)
        step_idx = 0
        last = None
        # progressive loop yields dict: {"sample": x_{t-1}, "pred_xstart": x0_pred}
        for out in self.diffusion.p_sample_loop_progressive(
            self.model,
            init_noise.shape,
            noise=init_noise,
            noise_sequence=noise_sequence,
            clip_denoised=False,
            model_kwargs=dict(y=y, x_cond=x_cond, rel_t=rel_t),
            device=self.device,
            progress=progress,
        ):
            step_idx += 1
            last = out
            if step_idx in probe_steps:
                # store float32 on GPU (caller may move/cast)
                probes[step_idx] = out["pred_xstart"].detach().float()
            if stop_at_max_probe and step_idx >= max_probe:
                break
        if last is None:
            raise RuntimeError("Diffusion sampling produced no outputs")
        return last["sample"], probes

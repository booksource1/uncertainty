import torch
import torchvision.transforms.functional as F
import numpy as np

# ImageNet Normalization constants
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

def denormalize(tensor):
    """反归一化: (T - mean) / std -> T * std + mean"""
    # tensor: [B, C, H, W] or [C, H, W]
    device = tensor.device
    mean = MEAN.to(device)
    std = STD.to(device)
    
    if tensor.ndim == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
        
    return tensor * std + mean

def normalize(tensor):
    """归一化"""
    device = tensor.device
    mean = MEAN.to(device)
    std = STD.to(device)
    
    if tensor.ndim == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
        
    return (tensor - mean) / std

class PerturbationManager:
    def __init__(self, device='cpu'):
        self.device = device

    def apply_perturbation(self, images, perturbation_type, severity=0, *, seed: int | None = None, seeds: torch.Tensor | None = None):
        """
        Args:
            images: normalized tensor, 支持
                - [B, C, H, W]
                - [B, T, C, H, W]（对每一帧独立施加相同类型扰动；随机采样因子按“每帧”独立）
            perturbation_type: 'none', 'noise', 'blur', 'blackout', 'photometric'
            severity: 1-5 (int) or float intensity
            seed: optional int seed to make perturbation deterministic for this call (batch-order dependent).
            seeds: optional per-sample seeds tensor shape [B] to make perturbation deterministic per sample (batch-order independent).
        """
        if perturbation_type == 'none' or severity == 0:
            return images
        if seed is not None and seeds is not None:
            raise ValueError("Provide only one of seed or seeds")

        # Denormalize to [0, 1] range for processing
        images_denorm = denormalize(images).clamp(0, 1)

        def _get_gen_for_sample(b: int, device: torch.device) -> torch.Generator | None:
            if seeds is not None:
                s = int(seeds[b].item())
                return torch.Generator(device=device).manual_seed(s)
            if seed is not None:
                # NOTE: batch-order dependent stream
                return torch.Generator(device=device).manual_seed(int(seed))
            return None
        
        if perturbation_type == 'noise':
            # Gaussian Noise
            sigma = 0.05 * severity # severity 1 -> 0.05, 5 -> 0.25
            if seeds is None and seed is None:
                noise = torch.randn_like(images_denorm) * sigma
                out = torch.clamp(images_denorm + noise, 0, 1)
            else:
                out = images_denorm.clone()
                if out.ndim == 5:
                    B, T, C, H, W = out.shape
                    for b in range(B):
                        g = _get_gen_for_sample(b, out.device)
                        n = torch.randn((T, C, H, W), device=out.device, generator=g) * sigma
                        out[b] = (out[b] + n).clamp(0, 1)
                else:
                    B, C, H, W = out.shape
                    for b in range(B):
                        g = _get_gen_for_sample(b, out.device)
                        n = torch.randn((C, H, W), device=out.device, generator=g) * sigma
                        out[b] = (out[b] + n).clamp(0, 1)
            
        elif perturbation_type == 'blur':
            # Gaussian Blur
            kernel_size = 2 * int(severity) + 1 # 3, 5, 7, 9, 11
            sigma = 0.5 * severity
            # torchvision gaussian_blur supports [B,C,H,W] (4D) but not [B,T,C,H,W] (5D)
            if images_denorm.ndim == 5:
                B, T, C, H, W = images_denorm.shape
                flat = images_denorm.reshape(B * T, C, H, W)
                flat_out = F.gaussian_blur(flat, kernel_size, [sigma, sigma])
                out = flat_out.reshape(B, T, C, H, W)
            else:
                out = F.gaussian_blur(images_denorm, kernel_size, [sigma, sigma])
            
        elif perturbation_type == 'blackout':
            # Random blackout patches
            out = images_denorm.clone()
            if images_denorm.ndim == 5:
                B, T, C, H, W = images_denorm.shape
                patch_size = max(1, int(min(H, W) * 0.1 * severity)) # 10% to 50% size
                for b in range(B):
                    # one mask per sample, shared across all context frames
                    g = _get_gen_for_sample(b, out.device)
                    if g is None:
                        y = int(np.random.randint(0, max(1, H - patch_size + 1)))
                        x = int(np.random.randint(0, max(1, W - patch_size + 1)))
                    else:
                        # generator `g` is created on `out.device` (often CUDA), so randint must use the same device.
                        y = int(torch.randint(0, max(1, H - patch_size + 1), (1,), device=out.device, generator=g).item())
                        x = int(torch.randint(0, max(1, W - patch_size + 1), (1,), device=out.device, generator=g).item())
                    out[b, :, :, y:y+patch_size, x:x+patch_size] = 0.0
            else:
                B, C, H, W = images_denorm.shape
                patch_size = max(1, int(min(H, W) * 0.1 * severity)) # 10% to 50% size
                for b in range(B):
                    g = _get_gen_for_sample(b, out.device)
                    if g is None:
                        y = int(np.random.randint(0, max(1, H - patch_size + 1)))
                        x = int(np.random.randint(0, max(1, W - patch_size + 1)))
                    else:
                        # generator `g` is created on `out.device` (often CUDA), so randint must use the same device.
                        y = int(torch.randint(0, max(1, H - patch_size + 1), (1,), device=out.device, generator=g).item())
                        x = int(torch.randint(0, max(1, W - patch_size + 1), (1,), device=out.device, generator=g).item())
                    out[b, :, y:y+patch_size, x:x+patch_size] = 0.0

        elif perturbation_type == 'photometric':
            # Photometric (Near-OOD): random brightness/contrast/saturation/hue/gamma jitter
            # All ops are applied in [0,1] space.
            s = float(severity)
            # factor ranges grow with severity
            # brightness/contrast around 1.0
            b_delta = 0.06 * s
            c_delta = 0.08 * s
            sat_delta = 0.10 * s
            hue_delta = 0.02 * s  # hue in [-0.5, 0.5]
            gamma_delta = 0.08 * s

            # sample one set of factors per sample; if [B,T,...], apply same factors to all T frames
            out = images_denorm.clone()
            if out.ndim == 5:
                B = out.shape[0]
                for b in range(B):
                    x = out[b]  # [T,C,H,W]
                    g = _get_gen_for_sample(b, x.device)
                    brightness = float((1.0 + (torch.rand(1, device=x.device, generator=g) * 2 - 1) * b_delta).clamp(0.1, 2.0).item())
                    contrast = float((1.0 + (torch.rand(1, device=x.device, generator=g) * 2 - 1) * c_delta).clamp(0.1, 3.0).item())
                    saturation = float((1.0 + (torch.rand(1, device=x.device, generator=g) * 2 - 1) * sat_delta).clamp(0.0, 3.0).item())
                    hue = float(((torch.rand(1, device=x.device, generator=g) * 2 - 1) * hue_delta).clamp(-0.5, 0.5).item())
                    gamma = float((1.0 + (torch.rand(1, device=x.device, generator=g) * 2 - 1) * gamma_delta).clamp(0.1, 3.0).item())

                    x = F.adjust_brightness(x, brightness)
                    x = F.adjust_contrast(x, contrast)
                    x = F.adjust_saturation(x, saturation)
                    x = F.adjust_hue(x, hue)
                    x = F.adjust_gamma(x, gamma, gain=1.0)
                    out[b] = x.clamp(0, 1)
            else:
                B = out.shape[0]
                for b in range(B):
                    x = out[b]  # [C,H,W]
                    g = _get_gen_for_sample(b, x.device)
                    brightness = float((1.0 + (torch.rand(1, device=x.device, generator=g) * 2 - 1) * b_delta).clamp(0.1, 2.0).item())
                    contrast = float((1.0 + (torch.rand(1, device=x.device, generator=g) * 2 - 1) * c_delta).clamp(0.1, 3.0).item())
                    saturation = float((1.0 + (torch.rand(1, device=x.device, generator=g) * 2 - 1) * sat_delta).clamp(0.0, 3.0).item())
                    hue = float(((torch.rand(1, device=x.device, generator=g) * 2 - 1) * hue_delta).clamp(-0.5, 0.5).item())
                    gamma = float((1.0 + (torch.rand(1, device=x.device, generator=g) * 2 - 1) * gamma_delta).clamp(0.1, 3.0).item())

                    x = F.adjust_brightness(x, brightness)
                    x = F.adjust_contrast(x, contrast)
                    x = F.adjust_saturation(x, saturation)
                    x = F.adjust_hue(x, hue)
                    x = F.adjust_gamma(x, gamma, gain=1.0)
                    out[b] = x.clamp(0, 1)
                
        else:
            print(f"Warning: Unknown perturbation {perturbation_type}")
            out = images_denorm

        # Re-normalize
        return normalize(out)


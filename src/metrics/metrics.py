import torch
import numpy as np
from PIL import Image

class UncertaintyMetrics:
    def __init__(self, device='cuda', use_lpips=True, use_dreamsim=True):
        self.device = device
        self.lpips_model = None
        self.dreamsim_model = None
        self.dreamsim_preprocess = None
        
        if use_lpips:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex').to(device)
                self.lpips_model.eval()
                print("LPIPS initialized.")
            except ImportError:
                print("LPIPS not installed.")
                
        if use_dreamsim:
            try:
                from dreamsim import dreamsim
                self.dreamsim_model, self.dreamsim_preprocess = dreamsim(pretrained=True, device=device)
                self.dreamsim_model.eval()
                print("DreamSim initialized.")
            except ImportError:
                print("DreamSim not installed.")
            except Exception as e:
                print(f"DreamSim init failed: {e}")

    def compute_pixel_variance(self, images_stack):
        """
        images_stack: [N, C, H, W]
        returns: mean pixel variance (scalar)
        """
        # model output may be bfloat16 under autocast; cast to float32 for stable stats
        images_stack = images_stack.float()
        # var over N dimension
        var_map = torch.var(images_stack, dim=0, unbiased=False)  # [C, H, W]
        return var_map.mean().item()

    def compute_diversity(self, images_stack):
        """
        images_stack: [N, C, H, W] (Assumed range [-1, 1])
        """
        # model output may be bfloat16 under autocast; cast to float32 for DreamSim/LPIPS and numpy conversion
        images_stack = images_stack.float()
        N = images_stack.shape[0]
        if N < 2: return {}
        
        lpips_dists = []
        dreamsim_dists = []
        
        # Prepare for DreamSim (needs PIL or specific tensor format)
        pil_images = []
        if self.dreamsim_model:
            # Denormalize [-1, 1] to [0, 255] for PIL
            imgs_np = ((images_stack.detach().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            imgs_np = imgs_np.transpose(0, 2, 3, 1) # [N, H, W, C]
            pil_images = [Image.fromarray(img) for img in imgs_np]

        # Prepare batched input for DreamSim to avoid overhead and potentially fix shape issues
        if self.dreamsim_model and pil_images:
            try:
                img0_list = []
                img1_list = []
                for i in range(N):
                    for j in range(i+1, N):
                        # Preprocess returns [1, 3, 224, 224] or [3, 224, 224]?
                        # Based on tests, it returns [1, 3, 224, 224].
                        p_img0 = self.dreamsim_preprocess(pil_images[i]).to(self.device)
                        p_img1 = self.dreamsim_preprocess(pil_images[j]).to(self.device)
                        img0_list.append(p_img0)
                        img1_list.append(p_img1)
                
                if img0_list:
                    # Handle dimension issues
                    if img0_list[0].ndim == 4:
                         # If [1, C, H, W], use cat to get [M, C, H, W]
                         img0_batch = torch.cat(img0_list, dim=0)
                         img1_batch = torch.cat(img1_list, dim=0)
                    else:
                         # If [C, H, W], use stack to get [M, C, H, W]
                         img0_batch = torch.stack(img0_list)
                         img1_batch = torch.stack(img1_list)
                    
                    # print(f"DEBUG: DreamSim input shape: {img0_batch.shape}")
                    dists = self.dreamsim_model(img0_batch, img1_batch)
                    
                    if isinstance(dists, torch.Tensor):
                        dreamsim_dists = dists.cpu().tolist() if dists.numel() > 1 else [dists.item()]
                    else:
                        dreamsim_dists = [float(dists)]
            except Exception as e:
                print(f"DreamSim batch calc failed: {e}")
                # Fallback or debug
                import traceback
                traceback.print_exc()

        for i in range(N):
            for j in range(i+1, N):
                # LPIPS
                if self.lpips_model:
                    # LPIPS expects [-1, 1]
                    dist = self.lpips_model(images_stack[i:i+1], images_stack[j:j+1])
                    lpips_dists.append(dist.item())
                    
        results = {}
        if lpips_dists:
            results['lpips'] = np.mean(lpips_dists)
        if dreamsim_dists:
            results['dreamsim'] = np.mean(dreamsim_dists)
            
        return results

    def compute_gt_error(self, images_stack: torch.Tensor, gt_image: torch.Tensor):
        """
        Expected error to ground truth (mean over seeds).

        Args:
            images_stack: [N, C, H, W] in range [-1, 1]
            gt_image: [C, H, W] in range [-1, 1]
        Returns:
            dict with keys:
              - lpips_gt: mean LPIPS(pred_i, gt)
              - dreamsim_gt: mean DreamSim(pred_i, gt)
              - mse_gt: mean MSE(pred_i, gt) in [0,1] space
              - psnr_gt: mean PSNR(pred_i, gt) computed from MSE in [0,1] space
              - ssim_gt: mean SSIM(pred_i, gt) in [0,1] space (standard SSIM, not MS-SSIM)
        """
        images_stack = images_stack.float()
        gt_image = gt_image.float()
        N = images_stack.shape[0]
        if N < 1:
            return {}

        results = {}

        # Pixel-space errors in [0,1] (more standard for MSE/PSNR/SSIM).
        # Convert [-1,1] -> [0,1]
        preds01 = ((images_stack + 1.0) * 0.5).clamp(0.0, 1.0)
        gt01 = ((gt_image + 1.0) * 0.5).clamp(0.0, 1.0).unsqueeze(0).expand(N, -1, -1, -1)

        # MSE
        mse_per = torch.mean((preds01 - gt01) ** 2, dim=(1, 2, 3))  # [N]
        results["mse_gt"] = float(mse_per.mean().item())

        # PSNR (in dB), using max_val=1.0
        eps = 1e-12
        psnr_per = 10.0 * torch.log10(1.0 / torch.clamp(mse_per, min=eps))
        results["psnr_gt"] = float(psnr_per.mean().item())

        # SSIM (batched, gaussian window)
        try:
            results["ssim_gt"] = float(_ssim_batch(preds01, gt01).mean().item())
        except Exception as e:
            print(f"SSIM gt calc failed: {e}")

        # LPIPS (batch)
        if self.lpips_model is not None:
            try:
                gt_rep = gt_image.unsqueeze(0).expand(N, -1, -1, -1)
                d = self.lpips_model(images_stack, gt_rep)  # [N,1,1,1] or [N,1]
                if isinstance(d, torch.Tensor):
                    results["lpips_gt"] = float(d.mean().item())
            except Exception as e:
                print(f"LPIPS gt calc failed: {e}")

        # DreamSim (batch)
        if self.dreamsim_model is not None and self.dreamsim_preprocess is not None:
            try:
                # Convert preds + gt to PIL for preprocess
                # [-1,1] -> [0,255]
                preds_np = ((images_stack.detach().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                preds_np = preds_np.transpose(0, 2, 3, 1)  # [N,H,W,C]
                pil_preds = [Image.fromarray(im) for im in preds_np]

                gt_np = ((gt_image.detach().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                gt_np = gt_np.transpose(1, 2, 0)  # [H,W,C]
                pil_gt = Image.fromarray(gt_np)

                # preprocess -> tensors on device
                pred_t = [self.dreamsim_preprocess(p).to(self.device) for p in pil_preds]
                gt_t = self.dreamsim_preprocess(pil_gt).to(self.device)

                if pred_t and pred_t[0].ndim == 4:
                    pred_batch = torch.cat(pred_t, dim=0)  # [N,3,224,224]
                else:
                    pred_batch = torch.stack(pred_t, dim=0)

                if gt_t.ndim == 4:
                    gt_batch = gt_t.expand(pred_batch.shape[0], -1, -1, -1)
                else:
                    gt_batch = gt_t.unsqueeze(0).expand(pred_batch.shape[0], -1, -1, -1)

                d = self.dreamsim_model(pred_batch, gt_batch)
                if isinstance(d, torch.Tensor):
                    results["dreamsim_gt"] = float(d.mean().item())
                else:
                    results["dreamsim_gt"] = float(d)
            except Exception as e:
                print(f"DreamSim gt calc failed: {e}")

        return results


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]
    return kernel2d


def _ssim_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    SSIM for batched RGB images in [0,1].
    Args:
      x, y: [N,3,H,W]
    Returns:
      ssim: [N]
    """
    if x.shape != y.shape:
        raise ValueError(f"SSIM expects same shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"SSIM expects [N,3,H,W], got {tuple(x.shape)}")

    device = x.device
    dtype = x.dtype
    ws = int(window_size)
    if ws % 2 == 0:
        ws += 1

    # Create depthwise gaussian window for 3 channels
    w = _gaussian_kernel(ws, sigma, device=device, dtype=dtype)  # [1,1,ws,ws]
    w = w.repeat(3, 1, 1, 1)  # [3,1,ws,ws]

    # Depthwise conv
    mu_x = torch.nn.functional.conv2d(x, w, padding=ws // 2, groups=3)
    mu_y = torch.nn.functional.conv2d(y, w, padding=ws // 2, groups=3)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = torch.nn.functional.conv2d(x * x, w, padding=ws // 2, groups=3) - mu_x2
    sigma_y2 = torch.nn.functional.conv2d(y * y, w, padding=ws // 2, groups=3) - mu_y2
    sigma_xy = torch.nn.functional.conv2d(x * y, w, padding=ws // 2, groups=3) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    # Average over C,H,W
    return ssim_map.mean(dim=(1, 2, 3))


import torch.nn.functional as F

window_size = 7
sigma = 1.5
# Create Gaussian kernel
coords = torch.arange(window_size).float() - window_size//2
gaussian_kernel = torch.exp(-(coords**2).unsqueeze(0) - (coords**2).unsqueeze(1) / (2*(sigma**2)))
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
gaussian_kernel = gaussian_kernel.view(1, 1, window_size, window_size)  # shape [1,1,7,7]
def ssim_loss(img, recon):
    C = img.shape[1]
    window = gaussian_kernel.to(img.device, img.dtype).repeat(C, 1, 1, 1)
    # Compute local means
    mu_img = F.conv2d(img, window, padding=window_size//2, groups=C)
    mu_rec = F.conv2d(recon, window, padding=window_size//2, groups=C)
    
    mu_img_sq = mu_img * mu_img
    mu_rec_sq = mu_rec * mu_rec
    mu_img_rec = mu_img * mu_rec
    sigma_img = F.conv2d(img * img, window, padding=window_size//2, groups=C) - mu_img_sq
    sigma_rec = F.conv2d(recon * recon, window, padding=window_size//2, groups=C) - mu_rec_sq
    sigma_img_rec = F.conv2d(img * recon, window, padding=window_size//2, groups=C) - mu_img_rec
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    
    ssim_n = (2 * mu_img_rec + C1) * (2 * sigma_img_rec + C2)
    ssim_d = (mu_img_sq + mu_rec_sq + C1) * (sigma_img + sigma_rec + C2)
    ssim_map = ssim_n / (ssim_d + 1e-8)
    loss = torch.clamp((1 - ssim_map) / 2, min=0, max=1)
    return loss.mean()

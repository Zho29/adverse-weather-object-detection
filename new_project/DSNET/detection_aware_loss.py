"""
Detection-Aware Loss Functions
Anti-over-smoothing losses:
1. Perceptual loss (VGG features)
2. Edge loss (Sobel gradients)
3. SSIM loss (structural similarity)
4. Reduced MSE weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class DetectionAwareLoss(nn.Module):
    """
    Combined loss for edge-preserving restoration
    
    Components:
    - Perceptual loss: Preserves textures and patterns
    - Edge loss: Preserves object boundaries
    - SSIM loss: Preserves structure
    - MSE loss: Pixel-level similarity (low weight)
    """
    
    def __init__(self, lambda_mse=0.2, lambda_perceptual=0.4, 
                 lambda_edge=0.3, lambda_ssim=0.1):
        super(DetectionAwareLoss, self).__init__()
        
        self.lambda_mse = lambda_mse
        self.lambda_perceptual = lambda_perceptual
        self.lambda_edge = lambda_edge
        self.lambda_ssim = lambda_ssim
        
        # VGG19 for perceptual loss (features from relu3_4)
        vgg = vgg19(pretrained=True).features[:23]  # Up to relu4_4
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def forward(self, restored, clear):
        """
        Args:
            restored: Restored image [B, 3, H, W]
            clear: Clear reference [B, 3, H, W]
            
        Returns:
            loss: Combined loss value
            loss_dict: Dictionary of individual losses
        """
        # 1. MSE Loss (reduced weight to avoid over-smoothing)
        L_mse = F.mse_loss(restored, clear)
        
        # 2. Perceptual Loss (VGG features)
        L_perceptual = self._perceptual_loss(restored, clear)
        
        # 3. Edge Loss (preserve boundaries)
        L_edge = self._edge_loss(restored, clear)
        
        # 4. SSIM Loss (structural similarity)
        L_ssim = self._ssim_loss(restored, clear)
        
        # Combined loss
        total_loss = (self.lambda_mse * L_mse +
                     self.lambda_perceptual * L_perceptual +
                     self.lambda_edge * L_edge +
                     self.lambda_ssim * L_ssim)
        
        loss_dict = {
            'total': total_loss.item(),
            'mse': L_mse.item(),
            'perceptual': L_perceptual.item(),
            'edge': L_edge.item(),
            'ssim': L_ssim.item()
        }
        
        return total_loss, loss_dict
    
    def _perceptual_loss(self, x, y):
        """Perceptual loss using VGG features"""
        # Denormalize from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        y = (y + 1) / 2
        
        # Get VGG features
        features_x = self.vgg(x)
        features_y = self.vgg(y)
        
        return F.mse_loss(features_x, features_y)
    
    def _edge_loss(self, x, y):
        """Edge loss using Sobel gradients"""
        # Apply Sobel filters (for each channel)
        sobel_x = self.sobel_x.repeat(3, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(3, 1, 1, 1)
        
        # Edges for restored
        edge_x_restored = F.conv2d(x, sobel_x, padding=1, groups=3)
        edge_y_restored = F.conv2d(x, sobel_y, padding=1, groups=3)
        edges_restored = torch.sqrt(edge_x_restored**2 + edge_y_restored**2 + 1e-6)
        
        # Edges for clear
        edge_x_clear = F.conv2d(y, sobel_x, padding=1, groups=3)
        edge_y_clear = F.conv2d(y, sobel_y, padding=1, groups=3)
        edges_clear = torch.sqrt(edge_x_clear**2 + edge_y_clear**2 + 1e-6)
        
        return F.l1_loss(edges_restored, edges_clear)
    
    def _ssim_loss(self, x, y, window_size=11):
        """SSIM loss for structural similarity"""
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.exp(-torch.arange(window_size, dtype=torch.float32)**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        
        window_1d = gauss.unsqueeze(1)
        window = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
        window = window.to(x.device)
        
        # Replicate window for each channel
        window = window.repeat(3, 1, 1, 1)
        
        # Constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.conv2d(x, window, padding=window_size//2, groups=3)
        mu_y = F.conv2d(y, window, padding=window_size//2, groups=3)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x * x, window, padding=window_size//2, groups=3) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=window_size//2, groups=3) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=window_size//2, groups=3) - mu_xy
        
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return 1 - ssim.mean()


def test_loss():
    """Test the detection-aware loss"""
    print("Testing Detection-Aware Loss")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create loss
    criterion = DetectionAwareLoss().to(device)
    
    # Test inputs
    restored = torch.randn(2, 3, 512, 512).to(device)
    clear = torch.randn(2, 3, 512, 512).to(device)
    
    print(f"\nInput shapes:")
    print(f"  Restored: {restored.shape}")
    print(f"  Clear: {clear.shape}")
    
    # Compute loss
    loss, loss_dict = criterion(restored, clear)
    
    print(f"\nLoss components:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value:.4f}")
    
    print(f"\nTotal loss: {loss.item():.4f}")
    
    # Test backward
    loss.backward()
    print("\n✓ Backward pass successful")
    
    print("\n" + "="*60)
    print("✓ Detection-Aware Loss test passed!")
    
    return criterion


if __name__ == '__main__':
    test_loss()

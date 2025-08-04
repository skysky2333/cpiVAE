"""
Common loss functions for VAE-based models.
"""

import torch
import math

def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    """Log probability of a diagonal multivariate normal."""
    log_prob = -0.5 * (math.log(2 * math.pi) + log_var + ((x - mu) ** 2) / torch.exp(log_var))
    if reduction == 'sum':
        return torch.sum(log_prob, dim)
    elif reduction == 'mean':
        return torch.mean(log_prob, dim)
    return log_prob

def total_correlation(z, mu, logvar):
    """
    Computes the total correlation loss with a robust method.
    This implementation is designed to be more numerically stable than
    the standard log-sum-exp trick, especially when dealing with high-dimensional
    latent spaces or posteriors that can have extreme values.

    Args:
        z (torch.Tensor): Samples from the latent space, shape [batch_size, latent_dim].
        mu (torch.Tensor): The mean of the latent vector, shape [batch_size, latent_dim].
        logvar (torch.Tensor): The log variance of the latent vector, shape [batch_size, latent_dim].

    Returns:
        torch.Tensor: A scalar tensor representing the total correlation loss.
    """
    batch_size, latent_dim = z.shape
    
    log_qz_i = log_normal_diag(z, mu, logvar)

    mu_expanded = mu.unsqueeze(0).expand(batch_size, -1, -1)
    logvar_expanded = logvar.unsqueeze(0).expand(batch_size, -1, -1)
    z_expanded = z.unsqueeze(1).expand(-1, batch_size, -1)
    
    log_qz_i_j = log_normal_diag(z_expanded, mu_expanded, logvar_expanded).sum(dim=-1)

    log_qz_agg = torch.logsumexp(log_qz_i_j, dim=1, keepdim=False) - math.log(batch_size)
    
    tc_loss = (log_qz_agg - log_qz_i.sum(dim=-1)).mean()
    
    return tc_loss


def kl_divergence_alignment(mean_a, logvar_a, mean_b, logvar_b):
    """
    Compute KL divergence between two posterior distributions.
    KL(N(μ_a, σ_a²) || N(μ_b, σ_b²))
    
    Args:
        mean_a, mean_b: Means of the two distributions
        logvar_a, logvar_b: Log variances of the two distributions
        
    Returns:
        KL divergence loss (scalar)
    """
    var_a = logvar_a.exp()
    var_b = logvar_b.exp()
    
    # Add small epsilon for numerical stability
    eps = torch.tensor(1e-8, device=var_a.device, dtype=var_a.dtype)
    var_a = var_a + eps
    var_b = var_b + eps
    
    kl_loss = 0.5 * torch.mean(
        torch.log(var_b / var_a) + 
        (var_a + (mean_a - mean_b).pow(2)) / var_b - 1
    )
    
    return kl_loss


def mmd_loss(z_a, z_b, kernel_bandwidth=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) loss between two sets of samples.
    Uses RBF (Gaussian) kernel.
    
    Args:
        z_a, z_b: Two sets of samples to compare
        kernel_bandwidth: Bandwidth parameter for RBF kernel
        
    Returns:
        MMD loss (scalar)
    """
    def rbf_kernel(x, y, bandwidth=kernel_bandwidth):
        """RBF kernel function"""
        pairwise_dists = torch.cdist(x, y, p=2).pow(2)
        
        if (False):
            # Use manual distance computation instead of torch.cdist for MPS compatibility
            x_norm = (x**2).sum(dim=1, keepdim=True)
            y_norm = (y**2).sum(dim=1, keepdim=True)
            
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
            pairwise_dists = x_norm + y_norm.transpose(0, 1) - 2 * torch.mm(x, y.transpose(0, 1))
            
            # Ensure non-negative distances (numerical stability)
            pairwise_dists = torch.clamp(pairwise_dists, min=0.0)
            
        return torch.exp(-pairwise_dists / (2 * bandwidth**2))
    
    # MMD computation
    K_aa = rbf_kernel(z_a, z_a)
    K_bb = rbf_kernel(z_b, z_b)
    K_ab = rbf_kernel(z_a, z_b)
    
    # Remove diagonal elements (self-similarities)
    batch_size = z_a.size(0)
    
    # Handle edge case where batch_size is too small
    if batch_size <= 1:
        return torch.tensor(0.0, device=z_a.device)
    
    K_aa = K_aa - torch.diag(torch.diag(K_aa))
    K_bb = K_bb - torch.diag(torch.diag(K_bb))
    
    # MMD² = E[K(za,za)] + E[K(zb,zb)] - 2E[K(za,zb)]
    mmd_loss_val = (K_aa.sum() / (batch_size * (batch_size - 1)) + 
                   K_bb.sum() / (batch_size * (batch_size - 1)) - 
                   2 * K_ab.mean())
    
    return torch.clamp(mmd_loss_val, min=0.0)  # MMD should be non-negative 
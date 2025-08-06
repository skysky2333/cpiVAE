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
    Computes the total correlation loss for FactorVAE.
    
    Total Correlation (TC) = KL[q(z) || ∏_j q(z_j)]
    where q(z) is the aggregated posterior and ∏_j q(z_j) is the product of marginals.
    
    This implementation uses a minibatch-based approximation where:
    - q(z) ≈ (1/N) Σ_i q(z|x_i) is approximated by the empirical distribution
    - q(z_j) ≈ (1/N) Σ_i q(z_j|x_i) is approximated marginally
    
    Args:
        z: Samples from the latent space [batch_size, latent_dim]
        mu: Mean of the latent distribution [batch_size, latent_dim]
        logvar: Log variance of the latent distribution [batch_size, latent_dim]
        
    Returns:
        TC loss (scalar)
    """
    batch_size, latent_dim = z.shape
    
    # Compute log q(z_i | x_i) for each sample under its own distribution
    # This gives us [batch_size, latent_dim] of log probabilities
    log_qz_given_x = log_normal_diag(z, mu, logvar, reduction=None)
    
    # For minibatch approximation of TC, we need to evaluate each z under all q(z|x)
    # Expand tensors for broadcasting
    z_expand = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
    mu_expand = mu.unsqueeze(0)  # [1, batch_size, latent_dim]
    logvar_expand = logvar.unsqueeze(0)  # [1, batch_size, latent_dim]
    
    # Compute log q(z_i | x_j) for all pairs (i,j)
    # Shape: [batch_size, batch_size, latent_dim]
    log_qz_i_given_x_j = log_normal_diag(z_expand, mu_expand, logvar_expand, reduction=None)
    
    # Compute log q(z) ≈ log (1/N) Σ_j q(z|x_j) using logsumexp for stability
    # Sum over the encoder index j, not the latent dimensions
    log_qz = torch.logsumexp(log_qz_i_given_x_j.sum(dim=2), dim=1) - math.log(batch_size)
    
    # Compute log ∏_j q(z_j) = Σ_j log q(z_j)
    # For each dimension j, compute log (1/N) Σ_i q(z_j|x_i)
    log_qz_j = torch.logsumexp(log_qz_i_given_x_j, dim=1) - math.log(batch_size)
    log_prod_qz_j = log_qz_j.sum(dim=1)
    
    # TC = E_q(z)[log q(z) - log ∏_j q(z_j)]
    tc = (log_qz - log_prod_qz_j).mean()
    
    return tc


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
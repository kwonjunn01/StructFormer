import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple

def compute_angular_dispersion(embeddings: torch.Tensor, tokens: torch.Tensor) -> float:
    """
    Compute the angular dispersion between patch embeddings and semantic tokens.
    
    Args:
        embeddings: Patch embeddings of shape (B, N, D) where B is batch size, N is number of patches, D is dimension
        tokens: Semantic tokens of shape (B, K, D) where K is number of tokens
    
    Returns:
        Angular dispersion value
    """
    # Normalize embeddings and tokens
    embeddings_norm = nn.functional.normalize(embeddings, p=2, dim=-1)  # (B, N, D)
    tokens_norm = nn.functional.normalize(tokens, p=2, dim=-1)  # (B, K, D)
    
    # Compute cosine similarities between all patch embeddings and tokens
    # (B, N, D) @ (B, D, K) -> (B, N, K)
    cosine_similarities = torch.bmm(embeddings_norm, tokens_norm.transpose(1, 2))
    
    # Convert to angles (arccos of cosine similarity)
    angles = torch.acos(torch.clamp(cosine_similarities, -1.0, 1.0))  # (B, N, K)
    
    # Compute angular dispersion (standard deviation of angles)
    # Flatten across batch and compute std
    angles_flat = angles.reshape(-1)
    dispersion = torch.std(angles_flat).item()
    
    return dispersion

def compute_angular_dispersion_per_token(embeddings: torch.Tensor, tokens: torch.Tensor) -> List[float]:
    """
    Compute the angular dispersion between patch embeddings and each semantic token separately.
    
    Args:
        embeddings: Patch embeddings of shape (B, N, D)
        tokens: Semantic tokens of shape (B, K, D)
    
    Returns:
        List of angular dispersion values for each token
    """
    dispersions = []
    
    # Normalize embeddings and tokens
    embeddings_norm = nn.functional.normalize(embeddings, p=2, dim=-1)  # (B, N, D)
    tokens_norm = nn.functional.normalize(tokens, p=2, dim=-1)  # (B, K, D)
    
    # Compute dispersion for each token
    for k in range(tokens.shape[1]):
        token_k = tokens_norm[:, k:k+1, :]  # (B, 1, D)
        
        # Compute cosine similarities
        cosine_sim = torch.bmm(embeddings_norm, token_k.transpose(1, 2))  # (B, N, 1)
        
        # Convert to angles
        angles = torch.acos(torch.clamp(cosine_sim, -1.0, 1.0))  # (B, N, 1)
        
        # Compute dispersion
        angles_flat = angles.reshape(-1)
        dispersion = torch.std(angles_flat).item()
        dispersions.append(dispersion)
    
    return dispersions

def plot_angular_dispersions(dispersions_dict: Dict[str, List[float]], save_path: str):
    """
    Plot angular dispersions across layers.
    
    Args:
        dispersions_dict: Dictionary with keys as layer names and values as dispersion values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Extract layer indices and values
    layers = sorted(dispersions_dict.keys(), key=lambda x: int(x.split('_')[1]))
    
    # Plot overall dispersion
    overall_dispersions = [dispersions_dict[layer][0] for layer in layers]
    plt.plot(range(len(layers)), overall_dispersions, 'b-o', linewidth=2, markersize=8, label='Overall Dispersion')
    
    # Plot per-token dispersions if available
    if len(dispersions_dict[layers[0]]) > 1:
        num_tokens = len(dispersions_dict[layers[0]]) - 1
        for token_idx in range(num_tokens):
            token_dispersions = [dispersions_dict[layer][token_idx + 1] for layer in layers]
            plt.plot(range(len(layers)), token_dispersions, '--', alpha=0.7, 
                    label=f'Token {token_idx + 1} Dispersion')
    
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Angular Dispersion (radians)', fontsize=14)
    plt.title('Angular Dispersion Between Patch Embeddings and Semantic Tokens', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(layers)), [f'Layer {i}' for i in range(len(layers))])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {save_path}")

def save_dispersions_to_file(dispersions_dict: Dict[str, List[float]], save_path: str):
    """
    Save dispersion values to a text file.
    
    Args:
        dispersions_dict: Dictionary with dispersion values
        save_path: Path to save the text file
    """
    with open(save_path, 'w') as f:
        f.write("Angular Dispersion Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        layers = sorted(dispersions_dict.keys(), key=lambda x: int(x.split('_')[1]))
        
        for layer in layers:
            dispersions = dispersions_dict[layer]
            f.write(f"{layer}:\n")
            f.write(f"  Overall Dispersion: {dispersions[0]:.4f}\n")
            
            if len(dispersions) > 1:
                for i, disp in enumerate(dispersions[1:]):
                    f.write(f"  Token {i+1} Dispersion: {disp:.4f}\n")
            f.write("\n")
    
    print(f"Dispersion values saved to {save_path}")
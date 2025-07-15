import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from models.structformer import StructFormer
from angular_dispersion import compute_angular_dispersion, compute_angular_dispersion_per_token, plot_angular_dispersions, save_dispersions_to_file

def main():
    # Model configuration based on crossformer_pp/small.yaml
    model_config = {
        'img_size': 224,
        'patch_size': [4, 8, 16, 32],
        'in_chans': 3,
        'num_classes': 1000,
        'embed_dim': 64,
        'depths': [2, 2, 18, 2],
        'num_heads': [2, 4, 8, 16],
        'group_size': [4, 4, 14, 7],
        'crs_interval': [4, 2, 1, 1],
        'mlp_ratio': [4.0, 4.0, 4.0, 4.0],
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.2,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
        'merge_size': [[2, 4], [2, 4], [2, 4]],
        'use_cpe': False,
        'group_type': 'constant',
        'pad_type': 0,
        'no_mask': False,
        'adaptive_interval': False,
        'use_acl': True
    }
    
    # Create model
    print("Creating model...")
    model = StructFormer(**model_config)
    model.cuda()
    model.eval()
    
    # Load checkpoint
    checkpoint_path = 'output/weight/debug/last.pth'
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights")
    
    # Create dummy input
    print("\nRunning angular dispersion analysis...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).cuda()
    
    save_dir = 'angular_dispersion_results'
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # Get layer outputs
        _, _, _, layer_outputs = model(dummy_input, return_layer_outputs=True)
        
        print(f"\nNumber of layers analyzed: {len(layer_outputs)}")
        
        # Process results
        dispersions = {}
        for layer_info in layer_outputs:
            layer_idx = layer_info['layer_idx']
            patch_embeddings = layer_info['patch_embeddings']
            semantic_tokens = layer_info['semantic_tokens']
            
            print(f"\nLayer {layer_idx}:")
            print(f"  Patch embeddings shape: {patch_embeddings.shape}")
            print(f"  Semantic tokens shape: {semantic_tokens.shape}")
            
            # Compute dispersions
            overall = compute_angular_dispersion(patch_embeddings, semantic_tokens)
            per_token = compute_angular_dispersion_per_token(patch_embeddings, semantic_tokens)
            
            layer_key = f'layer_{layer_idx}'
            dispersions[layer_key] = [overall] + per_token
            
            print(f"  Overall dispersion: {overall:.4f}")
            print(f"  Per-token dispersions: {[f'{d:.4f}' for d in per_token]}")
    
    # Save results
    plot_path = os.path.join(save_dir, 'angular_dispersion_plot.png')
    text_path = os.path.join(save_dir, 'angular_dispersion_values.txt')
    
    plot_angular_dispersions(dispersions, plot_path)
    save_dispersions_to_file(dispersions, text_path)
    
    print(f"\nResults saved to {save_dir}/")
    print("Done!")
    
    # Display results summary
    print("\n" + "="*50)
    print("ANGULAR DISPERSION SUMMARY")
    print("="*50)
    layers = sorted(dispersions.keys(), key=lambda x: int(x.split('_')[1]))
    for layer in layers:
        vals = dispersions[layer]
        print(f"{layer}: Overall={vals[0]:.4f}, Token1={vals[1]:.4f}, Token2={vals[2]:.4f}, Token3={vals[3]:.4f}")

if __name__ == '__main__':
    main()
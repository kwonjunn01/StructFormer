# StructFormer

This repository contains the implementation of StructFormer, a versatile vision transformer that leverages structured attention mechanisms with semantic tokens for improved visual representation learning.

## Overview

StructFormer introduces a novel approach to vision transformers by incorporating:
- **Structured Attention**: Cross-scale attention mechanism with semantic tokens
- **Angular Dispersion Analysis**: Measuring the relationship between patch embeddings and semantic tokens across layers
- **Efficient Multi-scale Processing**: Hierarchical architecture with adaptive group sizes

## Key Features

- Cross-scale attention with semantic tokens
- Dynamic position bias for improved spatial understanding
- Hierarchical architecture with 4 stages
- Angular dispersion measurement tools for model analysis
- Support for various model sizes (Small, Base, Large)

## Installation

```bash
# Clone the repository
git clone https://github.com/kwonjunn01/StructFormer.git
cd StructFormer

# Install dependencies
pip install -r requirements.txt
```

## Model Architecture

StructFormer uses a hierarchical architecture with:
- **Patch Embedding**: Multi-scale patch sizes [4, 8, 16, 32]
- **4 Stages**: With depths [2, 2, 18, 2] for the small variant
- **Semantic Tokens**: 3 learnable tokens that interact with patch embeddings
- **Group Attention**: Adaptive group sizes for efficient computation

## Usage

### Training

```bash
python -m torch.distributed.launch --nproc_per_node=4 \
    main.py \
    --cfg configs/structformer_pp/small.yaml \
    --data-path /path/to/imagenet \
    --batch-size 128
```

### Evaluation

```bash
python main.py \
    --cfg configs/structformer_pp/small.yaml \
    --data-path /path/to/imagenet \
    --batch-size 1 \
    --resume /path/to/checkpoint.pth \
    --eval
```

### Angular Dispersion Analysis

We provide tools to analyze the angular dispersion between patch embeddings and semantic tokens:

```bash
python test_angular_nodist.py
```

This will generate:
- `angular_dispersion_results/angular_dispersion_plot.png` - Visualization of dispersion across layers
- `angular_dispersion_results/angular_dispersion_values.txt` - Numerical results

## Model Zoo

| Model | Parameters | FLOPs | Top-1 Acc | Download |
|-------|------------|-------|-----------|----------|
| StructFormer-S | 23.3M | 4.4G | TBD | [link](#) |
| StructFormer-B | TBD | TBD | TBD | [link](#) |
| StructFormer-L | TBD | TBD | TBD | [link](#) |

## Angular Dispersion Results

Our analysis shows that angular dispersion between patch embeddings and semantic tokens increases with depth:
- **Early layers (0-1)**: Low dispersion (~0.05-0.07)
- **Middle layers (2)**: Sharp increase (~0.11)
- **Final layer (3)**: Highest overall dispersion (~0.15) but lower per-token dispersion

This pattern suggests the model learns increasingly specialized representations as depth increases.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{structformer2024,
  title={StructFormer: A Versatile Vision Transformer with Structured Attention},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

This implementation is based on the CrossFormer architecture and includes modifications for structured attention with semantic tokens.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
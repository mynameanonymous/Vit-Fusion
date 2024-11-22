
## Abstract
In computer vision tasks, combining multiple independently-trained Vision Transformers (ViTs)  can unlock greater generalization and robustness. However, the inherent complexities of ViTs—such as multi-head self-attention, layer normalization, and residual connections—require a tailored approach to model fusion. As a result, the conventional fusion techniques designed for simpler architectures are ill-suited for ViTs. Existing methods enforce full alignment across components, leading to excessive redundancy and computational inefficiency, particularly when fusing heterogeneous architectures of different scales. To overcome these challenges, we propose a novel multi-ViT fusion framework that combines conditional optimal transport (COT) for selective component alignment with structured pruning for targeted model compression. Our approach leverages COT and pruning to align only the most informative components (e.g., crucial attention heads and key normalization parameters) across models, bypassing unimportant elements and thus avoiding model bloat. Following this selective alignment, structured pruning is applied to eliminate redundant heads, neurons, and layers, further compressing the fused model and enhancing its efficiency. Our approach supports heterogeneous fusion for models of differing sizes, enhancing adaptability across diverse tasks. We demonstrate the strength of our fusion approach through extensive experiments, including the fusion and fine-tuning of ViTs on CIFAR10, CIFAR100, TINY IMAGENET, and IMAGENET-1K datasets. Our fused models consistently surpass the original converged models across tasks and datasets by approximately 1.0\%, while achieving significant reductions in computational and storage requirements. 
 

## How to run a Transformer Fusion experiment

1. **Clone the repository**

```
git clone https://github.com/graldij/transformer-fusion
```

2. **Using `Python 3.9`, install the required packages**

```
pip install -r requirements.txt
```

3.  **Download the zipped folder with the two Transformer models from this [link](https://drive.google.com/file/d/1ez2VqveQSJyBJ0WlzdrsFetoIruZU4Ph/view?usp=sharing) and extract the `models` folder into root of the repository.**

4. **Run the `main.py` script as follows:**
```
python main.py fuse_hf_vit_cifar10.yaml
```



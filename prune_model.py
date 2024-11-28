
import torch
from otfusion.utils import prune_and_fuse_layer
import torch.nn as nn



def prune_model(model, target_size=512):
    # """
    # Function to prune a transformer model.
    # - model: the transformer model to be pruned.
    # - target_size: the number of neurons to keep after pruning in each layer.
    # Access the transformer layers in ViTForImageClassification
    for i, layer in enumerate(model.vit.encoder.layer):
        # Prune the neurons in the dense layer inside each transformer block
        pruned_layer = prune_and_fuse_layer(layer.output.dense.weight, target_size)
        
        # Convert the pruned weights to torch.nn.Parameter
        model.vit.encoder.layer[i].output.dense.weight = nn.Parameter(pruned_layer)

    return model


import torch.nn as nn

def adjust_model_for_pruning(model, pruned_size, num_attention_heads):
    """
    Adjust the model's layers to match the pruned size and attention head sizes after pruning.
    pruned_size: The new size after pruning (e.g., 512)
    num_attention_heads: Number of attention heads (e.g., 12)
    """
    # Ensure that the pruned size is divisible by the number of attention heads
    if pruned_size % num_attention_heads != 0:
        raise ValueError(f"Pruned size {pruned_size} must be divisible by the number of attention heads {num_attention_heads}.")

    head_dim = pruned_size // num_attention_heads  # Dimension of each attention head

    # Loop over each transformer layer in the Vision Transformer
    for layer in model.vit.encoder.layer:
        # Adjust the self-attention mechanism
        layer.attention.attention.query = nn.Linear(384, pruned_size)
        layer.attention.attention.key = nn.Linear(384, pruned_size)
        layer.attention.attention.value = nn.Linear(384, pruned_size)
        layer.attention.output.dense = nn.Linear(pruned_size, pruned_size)

        # Adjust the number of attention heads
        layer.attention.attention.num_attention_heads = num_attention_heads
        layer.attention.attention.attention_head_size = head_dim

        # Adjust the feed-forward network (FFN) sizes
        layer.intermediate.dense = nn.Linear(pruned_size, pruned_size)
        layer.output.dense = nn.Linear(pruned_size, 384)

    # Adjust the final layer normalization and projection (if necessary)
    model.vit.layernorm = nn.LayerNorm(pruned_size)
    model.vit.pooler = nn.Linear(pruned_size, model.config.hidden_size)

    return model

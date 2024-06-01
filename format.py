import jax
import jax.numpy as jnp
import numpy as np
import json
import tensorflow as tf
import torch
from PIL import Image
import io
import importlib
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import jax.numpy as jnp
import torch

def pytree_display(example: dict):
    def print_shape_or_value(x):
        if isinstance(x, (jnp.ndarray, np.ndarray, tf.Tensor)):
            return f"Shape: {x.shape}"
        else:
            return x
    def apply_to_nested_dict(func, d):
        if isinstance(d, dict):
            return {k: apply_to_nested_dict(func, v) for k, v in d.items()}
        else:
            return func(d)
    converted_tree = jax.tree_util.tree_map(print_shape_or_value, example)
    formatted_output = json.dumps(converted_tree, indent=4)
    print(formatted_output)
    
def pytree_save(example: dict, filename: str):
    def print_shape_or_value(x):
        if isinstance(x, (jnp.ndarray, np.ndarray, tf.Tensor)):
            return f"Shape: {x.shape}"
        else:
            return x
    def apply_to_nested_dict(func, d):
        if isinstance(d, dict):
            return {k: apply_to_nested_dict(func, v) for k, v in d.items()}
        else:
            return func(d)
    converted_tree = jax.tree_util.tree_map(print_shape_or_value, example)
    formatted_output = json.dumps(converted_tree, indent=4)
    with open(filename, 'w') as f:
        f.write(formatted_output)

def dataset_display(dataset: tf.data.Dataset):
    for step in dataset.take(1):
        pytree_display(step)
        
def standardize_pytree(params):
    def print_shape_or_value(x):
        if isinstance(x, (jnp.ndarray, np.ndarray, tf.Tensor)):
            return f'Shape: {x.shape}'
        else:
            return x
    def apply_to_nested_dict(func, d):
        if isinstance(d, dict):
            return {k: apply_to_nested_dict(func, v) for k, v in d.items()}
        else:
            return func(d)
    converted_tree = jax.tree_util.tree_map(print_shape_or_value, params)
    formatted_output = json.dumps(converted_tree, indent=4)
    return formatted_output

def contain_nan(example: Union[dict, np.ndarray, jnp.ndarray]):
    def has_nan(x):
        if isinstance(x, np.ndarray):
            return np.isnan(x).any()
        elif isinstance(x, jnp.ndarray):
            return jnp.isnan(x).any()
        else:
            return False
    
    if isinstance(example, dict):
        return any(has_nan(x) for x in jax.tree_util.tree_leaves(example))
    elif isinstance(example, (np.ndarray, jnp.ndarray)):
        return has_nan(example)
    else:
        raise ValueError(f"Unsupported type: {type(example)}")

# test_example = {
#     'a': np.array([1, 2, 3]),
#     'b': {
#         'c': np.array([np.nan, 2, 3]),
#         'd': np.array([1, 2, 3]),
#     }
# }

# print(contain_nan(test_example))

def save_attention_mask(matrix, filename="attn_mask.png"):
    if isinstance(matrix, np.ndarray):
        data = matrix
    elif isinstance(matrix, jnp.ndarray):
        data = np.array(matrix)
    elif torch.is_tensor(matrix):
        data = matrix.numpy()
    else:
        raise ValueError("Input must be a numpy array, jax array, or tensor.")

    # Ensure the matrix is of the correct shape
    assert data.shape[0] == data.shape[1], "Input matrix must be square (A, A)."
    
    A = data.shape[0]

    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='gray', interpolation='none')
    plt.colorbar()
    plt.title("Attention Mask")

    # Set the labels for the x and y axis
    plt.xlabel("Columns")
    plt.ylabel("Rows")

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()
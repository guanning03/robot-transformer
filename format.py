import jax
import jax.numpy as jnp
import numpy as np
import json
import tensorflow as tf

from PIL import Image
import io
import importlib

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
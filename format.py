import jax
import jax.numpy as jnp
import numpy as np
import json
import tensorflow as tf

from PIL import Image
import io
import importlib
from typing import Optional, Union

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
import os

from typing import Any, Callable, Dict, Optional, Sequence, Union, NamedTuple, Tuple

import copy
import enum
import flax
import flax.linen as nn
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import rlds
import reverb
from rlds import transformations
import tensorflow_datasets as tfds
import tree
from format import pytree_display, dataset_display, standardize_pytree, contain_nan
import wandb

import abc
import dataclasses
import math
from typing import Dict, Optional
import json
from rlds import rlds_types
import tensorflow as tf
from PIL import Image
from IPython import display
import tensorflow_datasets as tfds
import functools
from typing import Callable, Sequence
import matplotlib.pyplot as plt
from rt1 import RT1, detokenize_action, tokenize_action
from load_data import get_file_list, load_data_from_hdf5
import sys
import pdb
import time
from flax.training import checkpoints

SEQUENCE_LENGTH = 15
NUM_ACTION_TOKENS = 15
LAYER_SIZE = 256
VOCAB_SIZE = 512
NUM_IMAGE_TOKENS = 81

rt1x_model = RT1(
    num_image_tokens=NUM_IMAGE_TOKENS,
    num_action_tokens=NUM_ACTION_TOKENS,
    layer_size=LAYER_SIZE,
    vocab_size=VOCAB_SIZE,
    # Use token learner to reduce tokens per image to 81.
    use_token_learner=True,
    # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
    world_vector_range=(-2.0, 2.0)
)

NUM_TOKENS_TOTAL = SEQUENCE_LENGTH * (NUM_IMAGE_TOKENS + NUM_ACTION_TOKENS)

def load_from_pretrained_straightforward(checkpoint_path):
    state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)
    variables = {
        'params': state_dict['params'],
        'batch_stats': state_dict['batch_stats'],
    }
    return variables

text_embeddings = json.load(open("text_embeddings.json", "r"))
test_iter = load_data_from_hdf5(file_list = ['./data/take_out_tissue/episode_1.hdf5'],
                                batch_size = 1,
                                file_batch_size = 1,
                                embedding_dict=text_embeddings)
batch = next(test_iter)

variables = load_from_pretrained_straightforward("checkpoints/test/checkpoint_465999")

model_output = rt1x_model.apply(
    variables,
    batch['observation'],
    jax.tree_map(lambda x: jnp.ones_like(x), batch['action']),
    train = False,
    rngs = {'random': jax.random.PRNGKey(0)}
)

# 将模型输出reshape为需要的形状
output_logits = jnp.reshape(
    model_output, (1, SEQUENCE_LENGTH, NUM_IMAGE_TOKENS + NUM_ACTION_TOKENS, -1)
)

# 初始化action_pred的列表
action_preds = []

# 遍历序列长度，计算每个时间步的动作预测
for t in range(SEQUENCE_LENGTH):
    action_logits = output_logits[:, t, ...]
    action_logits = action_logits[:, NUM_IMAGE_TOKENS - 1 : -1]
    
    action_logp = jax.nn.softmax(action_logits)
    action_token = jnp.argmax(action_logp, axis=-1)
    
    action_detokenized = detokenize_action(action_token, VOCAB_SIZE, world_vector_range=(-2.0, 2.0))
    action_preds.append(action_detokenized)

# 打开文件，写入原始动作和解码后的动作，并计算MSE
mse_total = 0
mse_count = 0

with open('action_predictions.txt', 'w') as f:
    for t in range(SEQUENCE_LENGTH):
        f.write(f"\nTime step {t}:\n")
        for key in batch["action"]:
            pred_value = action_preds[t][key]
            gt_value = batch["action"][key][0][t]
            f.write(f"{key}: pred {pred_value}, gt {gt_value}\n")
            
            # 计算每个键的MSE
            if key != 'terminate_episode':  # 跳过terminate_episode，因为它是分类而不是回归
                mse = jnp.mean((pred_value - gt_value) ** 2)
                if not jnp.equal(gt_value, jnp.zeros_like(gt_value)):
                    mse_total += mse
                    mse_count += 1

# 计算总体的MSE
mse_average = mse_total / mse_count if mse_count > 0 else 0

print(f"Mean Square Error (MSE): {mse_average}")
with open('action_predictions.txt', 'a') as f:
    f.write(f"\nMean Square Error (MSE): {mse_average}")
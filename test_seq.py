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

variables = load_from_pretrained_straightforward("checkpoints/test/checkpoint_465999")

### TODO: make a batch as the model input
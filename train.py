import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

# @title Batch, and sample one training sample

MODE = 'pretrain' # 'finetune' or 'pretrain'
wandb_config = {
  'login_api_key': '256879fdda25bc1fb8ee4f0310e71615e92f75c9',
  'project': 'rt-1-x',
  'name': f'{MODE}',
  'disabled': True
}

current_time = time.strftime("%Y%m%d-%H%M%S")

wandb.login(key=wandb_config['login_api_key'])
wandb.init(project=wandb_config['project'], name=wandb_config['name'], mode = 'online' if wandb_config['disabled'] == False else 'disabled')

# Larger shuffle buffer leads to better performance, but consumes more RAM
datasets = []
weights = []

# for name, dataset in DATASET_NAME_TO_TRAJECTORY_DATASET.items():

#   datasets.append(dataset.shuffle(10))
#   weights.append(float(DATASET_NAME_TO_WEIGHTS[name]))

# dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=weights)

# # Larger shuffle buffer leads to better performance, but consumes more RAM
# dataset = dataset.shuffle(1)

# dataset = dataset.batch(BATCH_SIZE)

# trajectory_dataset_iter = iter(dataset)

# sample = next(trajectory_dataset_iter)


SEQUENCE_LENGTH = 32
NUM_ACTION_TOKENS = 15
LAYER_SIZE = 256
VOCAB_SIZE = 512
NUM_IMAGE_TOKENS = 81
NUM_IMAGES = 3

rt1x_model = RT1(
    num_image_tokens=NUM_IMAGE_TOKENS,
    num_action_tokens=NUM_ACTION_TOKENS,
    layer_size=LAYER_SIZE,
    vocab_size=VOCAB_SIZE,
    # Use token learner to reduce tokens per image to 81.
    use_token_learner=True,
    # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
    world_vector_range=(-2.0, 2.0),
    num_images = NUM_IMAGES
)

NUM_TOKENS_TOTAL = SEQUENCE_LENGTH * (NUM_IMAGE_TOKENS * NUM_IMAGES + NUM_ACTION_TOKENS)

# Initialize random weights for the model and run a forward pass.
obs = {
    "image": jnp.ones((1, SEQUENCE_LENGTH, 300, 300, 3)),
    "natural_language_embedding": jnp.ones((1, SEQUENCE_LENGTH, 512)),
}
act = {
    # "world_vector": jnp.ones((1, 15, 3)),
    # "rotation_delta": jnp.ones((1, 15, 3)),
    # "gripper_closedness_action": jnp.ones((1, 15, 1)),
    # "base_displacement_vertical_rotation": jnp.ones((1, 15, 1)),
    # "base_displacement_vector": jnp.ones((1, 15, 2)),
    # "terminate_episode": jnp.ones((1, 15, 3), dtype=jnp.int32),
    'arms_l0': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'arms_l1': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'arms_l2': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'arms_l3': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'arms_l4': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'arms_l5': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'arms_l6': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'arms_r0': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'arms_r1': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'arms_r2': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'arms_r3': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'arms_r4': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'arms_r5': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'arms_r6': jnp.ones((1, SEQUENCE_LENGTH, 1), dtype=jnp.int32),
    'terminate_episode': jnp.ones((1, SEQUENCE_LENGTH, 3), dtype=jnp.int32),
}

### 把from scratch改成from pretrained
scratch_variables = rt1x_model.init(
    {
        "params": jax.random.PRNGKey(0),
        "random": jax.random.PRNGKey(0),
    },
    obs,
    act,
    train=False,
)

def load_from_pretrained(checkpoint_path):
    state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)
    variables = {
        'params': state_dict['params'],
        'batch_stats': state_dict['batch_stats'],
    }

    ### 对positional_embedding做特殊修改
    variables['params']['Transformer_0']['Dense_1'] = scratch_variables['params']['Transformer_0']['Dense_1'].copy()
    
    return variables

pretrained_variables = load_from_pretrained('rt_1_x_jax')

if MODE == 'pretrain':
  decision_variables = scratch_variables
  del pretrained_variables
elif MODE == 'finetune':
  decision_variables = pretrained_variables
  del scratch_variables

model_output = rt1x_model.apply(
    decision_variables,
    obs,
    act,
    train=False,
    rngs={"random": jax.random.PRNGKey(0)},
) # shape: [1, 1380, 512] (1, 15 * (81 + 11), 512)


# Inspect the model weights and output.

param_count = sum(x.size for x in jax.tree_util.tree_leaves(decision_variables["params"]))
print(f"Number of parameters: {param_count}")

print(f"Output shape: {model_output.shape}.")

# Extract the actions from the model.
time_step_tokens = (
    NUM_IMAGE_TOKENS * NUM_IMAGES + NUM_ACTION_TOKENS
)
output_logits = jnp.reshape(
    model_output, (1, SEQUENCE_LENGTH, time_step_tokens, -1)
)
action_logits = output_logits[:, -1, ...]
action_logits = action_logits[:, NUM_IMAGE_TOKENS * NUM_IMAGES - 1 : -1]

action_logp = jax.nn.softmax(action_logits)
action_token = jnp.argmax(action_logp, axis=-1)

action_detokenized = detokenize_action(action_token, VOCAB_SIZE, world_vector_range=(-2.0, 2.0))
print(f"Detokenized actions: {action_detokenized}")

# @title Additional data preprocessing

def _is_not_terminal(trajectory):
  # -1 selects the final step in a trajectory
  if trajectory[rlds.IS_TERMINAL][-1]:
    return False
  return True


def convert_dtype_and_crop_images(
    images,
    resize_size,
    training: bool = True,
    convert_dtype: bool = True,
    seed: Optional[tf.Tensor] = None,
):
  """Convert uint8 images to float32 and square crop.

  Args:
    images: [B, H, W, 3] uint8 tensor of images.
    resize_size: (H, W) of resize.
    training: If we are in training (random crop) or not-training (fixed crop).
    convert_dtype: whether or not to convert the image to float32 in the range
      of (0, 1).
    seed: Optional seed of shape (2,) for giving to tf.random.stateless_uniform

  Returns:
    [B, crop_size, crop_size, 3] images of dtype float32.
  """

  if seed is None:
    seed = tf.random.uniform(shape=(2,), maxval=2**30, dtype=tf.int32)

  seed2 = tf.random.experimental.stateless_split(seed, num=1)[0]

  if convert_dtype:
    images = tf.image.convert_image_dtype(images, tf.float32)
  image_height = images.get_shape().as_list()[-3]
  image_width = images.get_shape().as_list()[-2]

  if training:
    if image_height == 512:
      ud_pad = 40
      lr_pad = 100
    elif image_height == 256:
      ud_pad = 20
      lr_pad = 50
    else:
      raise ValueError(
          'convert_dtype_and_crop_images only supports image height 512 or 256.'
      )
    max_y = 2 * ud_pad
    max_x = 2 * lr_pad
    images = tf.image.pad_to_bounding_box(
        images,
        offset_height=ud_pad,
        offset_width=lr_pad,
        target_height=image_height + 2 * ud_pad,
        target_width=image_width + 2 * lr_pad,
    )
    offset_y = tf.random.stateless_uniform(
        (), maxval=max_y + 1, dtype=tf.int32, seed=seed
    )
    offset_x = tf.random.stateless_uniform(
        (), maxval=max_x + 1, dtype=tf.int32, seed=seed2
    )
    images = tf.image.crop_to_bounding_box(
        images, offset_y, offset_x, image_height, image_width
    )

  # Add resize in pipeline for jax.
  images = tf.image.resize(images, size=resize_size)
  return images


def prepare_for_model_input(
    ds, target_height, target_width, training
):
  """Removes terminal trajectory, string from features and crops image."""
  ds = ds.filter(_is_not_terminal)

  # Remove non-jax types.
  def _remove_str(steps):
    if 'natural_language_instruction' in steps['observation']:
      del steps['observation']['natural_language_instruction']
    return steps

  ds = ds.map(_remove_str, num_parallel_calls=tf.data.AUTOTUNE)

  # Cropping augmentation.
  def _add_crop_augmentation(step):
    # Crop and pad augmentation. Added for jax.
    image = step['observation']['image']
    step['observation']['image'] = (
        convert_dtype_and_crop_images(
            image, (target_height, target_width), training=training
        )
    )
    return step

  ds = ds.map(_add_crop_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

  return ds

# @title Set up sharding and data parallel mesh

# Actual global batch size is 1024. Use a smaller batch size for this colab
# example.
PER_DEVICE_BATCH_SIZE = 1

def reshard(tree, shardings):
  """Take an arbitrarily sharded pytree and shard it according to `shardings`.

  From `big_vision.utils.reshard`. See that doc for full details.

  Args:
    tree: a pytree of arrays.
    shardings: a (prefix) pytree of jax array shardings.

  Returns:
    A pytree of global jax arrays that follows provided shardings.
  """

  def _make_global_arr(x, shard, shape):
    # Avoid unnecessary copies and transfers:
    if hasattr(x, "sharding") and x.sharding.is_equivalent_to(
        shard, len(shape)
    ):  # pylint: disable=line-too-long
      return x
    if not getattr(x, "is_fully_addressable", True):
      raise RuntimeError(
          "Trying to reshard a non-fully-addressable array. "
          "Please see the doc-comment for detailed explanation."
      )
    x = jax.device_get(x)  # Might be on local devices.
    xs = [
        jax.device_put(x[s], device=d)
        for d, s in shard.addressable_devices_indices_map(shape).items()
    ]
    return jax.make_array_from_single_device_arrays(shape, shard, xs)

  shapes = jax.tree_map(np.shape, tree)
  shardings = tree_broadcast(shardings, tree)
  return jax.tree_map(_make_global_arr, tree, shardings, shapes)

def tree_broadcast(prefix, target):
  """Broadcasts a prefix tree to a full tree.

  See big_vision.utils.tree_broadcast.

  Args:
    prefix: prefix pytree.
    target: boradcast target for a prefix tree.

  Returns:
    prefix tree broadcasted to a target tree.
  """

  def _broadcast(leaf, subtree):
    return jax.tree_map(lambda _: leaf, subtree)

  return jax.tree_map(_broadcast, prefix, target)


NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec

# train_dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=weights)

# train_dataset = prepare_for_model_input(
#     train_dataset, target_height=300, target_width=300, training=True
# )

# Creating mesh and shardings.
num_devices = len(jax.devices())
mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((num_devices,)), ("data",)
)

# Data parallel mesh.
sharding = jax.sharding.NamedSharding(mesh, P("data"))
replicate_sharding = NamedSharding(mesh, P())

global_batch_size = jax.device_count() * PER_DEVICE_BATCH_SIZE
local_batch_size = jax.local_device_count() * PER_DEVICE_BATCH_SIZE
# train_dataset = train_dataset.batch(local_batch_size, drop_remainder=True)

# train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

file_list = get_file_list("data")

text_embeddings = json.load(open("text_embeddings.json", "r"))
train_iter = load_data_from_hdf5(file_list, batch_size=global_batch_size, file_batch_size=global_batch_size // 1, 
                                 embedding_dict=text_embeddings, max_length = SEQUENCE_LENGTH)

# train_iter = train_dataset.as_numpy_iterator()

sample_batch = jax.tree_map(lambda x: x, next(train_iter))

pytree_display(sample_batch)

### TODO: 在这个位置插入一个train_iter，要和sample_batch相同

print(f"Local batch size: {local_batch_size}")
print(f"Global batch size: {global_batch_size}")
print(f"Devices: {jax.devices()}")
print(f"Sample batch keys: {sample_batch.keys()}")

# @title Create the train init fn, train step fn, and loss function.

@flax.struct.dataclass
class TrainState:
  step: int
  params: Any
  opt_state: optax.OptState
  batch_stats: Any


def create_train_state(model, batch, rng, optimizer):
    """Creates the train state and initial metrics for agent."""
    obs_input = batch["observation"]
    act_input = batch["action"]

    rng, rng2, rng3 = jax.random.split(rng, 3)
    
    ### 将这个改成预训练的模型
    
    # variables = model.init(
    #     {"params": rng, "random": rng3},
    #     obs=obs_input,
    #     act=act_input,
    #     train=False,
    # )

    params = flax.core.unfreeze(decision_variables["params"])
    batch_stats = flax.core.unfreeze(decision_variables["batch_stats"])

    train_state = TrainState(
        step=0,
        params=flax.core.unfreeze(params),
        opt_state=optimizer.init(params),
        batch_stats=batch_stats,
    )
    return train_state


def train(batch, state, model, optimizer, rng):
  """Performs a single training step."""
  rng, loss_rng = jax.random.split(rng)

  def loss_fn(params):
    variables = {"params": params, "batch_stats": state.batch_stats}
    per_example_loss, new_variables = rt1_loss(
        model, batch=batch, variables=variables, rng=loss_rng
    )
    loss = jnp.mean(per_example_loss)
    return loss, new_variables["batch_stats"]

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, new_batch_stats), grad = grad_fn(state.params)

  loss = jnp.mean(loss)

  updates, new_opt_state = optimizer.update(
      grad, state.opt_state, state.params
  )

  new_params = optax.apply_updates(state.params, updates)
  new_state = state.replace(
      step=state.step + 1,
      params=flax.core.unfreeze(new_params),
      opt_state=flax.core.unfreeze(new_opt_state),
      batch_stats=flax.core.unfreeze(new_batch_stats),
  )

  metrics_update = {
      "loss": loss,
  }
  return new_state, metrics_update


def rt1_loss(
      model,
      batch,
      variables,
      rng,
  ):
  """Implements the RT-1 loss."""
  observation = batch["observation"]
  action = batch["action"]

  bs = observation["image"].shape[0]
  seqlen = observation["image"].shape[1]

  # First, we encode the observations using the model.encode method.
  # This will give us an observation encoding (for the entire sequence).
  rng, params_rng = jax.random.split(rng)
  rng, dropout_rng = jax.random.split(rng)
  rng, sd_rng = jax.random.split(rng)
  rng, random_rng = jax.random.split(rng)
  logits, new_variables = model.apply(
      variables,
      obs=observation,
      act=action,
      train=True,
      mutable=["batch_stats"],
      rngs={
          "params": params_rng,
          "dropout": dropout_rng,
          "random": random_rng,
      },
  )
  
  vocab_size = model.vocab_size

  # `action` is dict of (B, T, ...), we combine actions into B*T batch to
  # tokenize.
  action = jax.tree_map(lambda x: jnp.reshape(x, (bs * seqlen, -1)), action)
  labels = tokenize_action(action, vocab_size=vocab_size)
  labels = jax.tree_map(lambda x: jnp.reshape(x, (bs, seqlen, -1)), labels)
  labels = labels[:, :, :, None]  # (B, 15, 15, 1)

  # Get num_action_tokens tokens for the action prediction. By default,
  # RT-1 computes the loss for all `seqlen * num_action_tokens`, not just
  # the final timestep's action.
  # In the default RT-1 setup (8 img, 11 act tokens), we have:
  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
  # |-----image tokens------|-------------action tokens--------------|
  #                      |----------------logits------------------|
  # For each time step, we want [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] for
  # the logits, for the "next token" prediction.
  num_image_tokens = model.num_image_tokens
  num_action_tokens = model.num_action_tokens
  num_images = model.num_images
  time_step_tokens = num_image_tokens * num_images + num_action_tokens
  logits = jnp.reshape(logits, (bs, seqlen, time_step_tokens, vocab_size))  ### (B, 15, 96, 512)
  logits = logits[:, :, num_image_tokens * num_images:] ### (B, 15, 15, 512)

  logp = jax.nn.log_softmax(logits)
  
  loglik = jnp.take_along_axis(logp, labels, axis=-1)
  loglik = jnp.mean(loglik, axis=(1, 2, 3))

  return -loglik, new_variables

# @title Set up the functions for training

optimizer = optax.adam(learning_rate=1e-4, eps=1e-7)

# Create the train state.
# input: batch, rng, ds_info
# output: state
agent_create_train_state = functools.partial(
    create_train_state, model=rt1x_model, optimizer=optimizer
)
create_train_state_jit = jax.jit(
    agent_create_train_state,
    out_shardings=replicate_sharding,
)

global_data_shape = jax.tree_map(
    lambda x: (global_batch_size,) + x.shape[1:], sample_batch
)

local_devices = mesh.local_devices
local_device_count = jax.local_device_count()

def _put_to_devices(x):
  per_device_arrays = np.split(x, local_device_count, axis=0)
  return jax.device_put(per_device_arrays, local_devices)

def _form_gda(local_data, global_shape):
  arrays = _put_to_devices(local_data)
  return jax.make_array_from_single_device_arrays(
      global_shape, sharding, arrays
  )

rng = jax.random.PRNGKey(0)

sample_batch = jax.tree_map(_form_gda, sample_batch, global_data_shape)
rng, agent_rng = jax.random.split(rng)
state = create_train_state_jit(
    batch=sample_batch, rng=agent_rng
)

# Create the train step.
agent_train = functools.partial(train, model=rt1x_model, optimizer=optimizer)
jitted_train_step = jax.jit(
    agent_train,
    out_shardings=(replicate_sharding, replicate_sharding),
)

# @title Run the train loop

num_train_steps = 10_000_000  # 1k for example, actual should be > 1M
log_loss_every_steps = 1
save_every_steps = 10000


# The state should be resharded since we may have loaded pretrained weights
# that need to be converted to jax.Arrays.
state_repl = reshard(state, shardings=replicate_sharding)
# The RNG must be replicated.
rng_repl = reshard(rng, shardings=replicate_sharding)

for step in range(num_train_steps):
  is_last_step = step == num_train_steps

  rng_repl = jax.random.fold_in(rng_repl, step)

  batch = next(train_iter)
  batch = jax.tree_map(_form_gda, batch, global_data_shape)
    
  state_repl, metrics_update = jitted_train_step(
      state=state_repl, batch=batch, rng=rng_repl
  )

  if (step + 1) % log_loss_every_steps == 0 or is_last_step:
    metrics_update = jax.device_get(metrics_update)
    print(f"Metrics: step={step}, {metrics_update}")
    wandb.log(metrics_update, step=step)
    
  if (step + 1) % save_every_steps == 0 or is_last_step:
    state_dict = {
        'params': flax.core.freeze(state_repl.params),
        'batch_stats': state_repl.batch_stats,
    }
    checkpoints.save_checkpoint(ckpt_dir = os.path.abspath(f'./checkpoints/{current_time}'), 
                                target = state_dict, 
                                step = step,
                                prefix = 'checkpoint_',
                                keep = None)

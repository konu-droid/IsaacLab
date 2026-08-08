# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom neural network models for the Lerobot Ts-Pnp task.

This module defines hand-written ``torch.nn.Module`` based skrl models so the user
has full control over the network architecture (instead of relying on skrl's YAML
model-instantiator). Two models are provided:

* :class:`CustomGaussianPolicy`  - a stochastic (Gaussian) policy used by PPO.
* :class:`CustomDeterministicValue` - a deterministic state-value critic.

Both share the same configurable MLP trunk built by :func:`build_mlp`. The trunk uses
``LayerNorm`` between linear layers, which is the main architectural difference from
skrl's default MLP and demonstrates that the architecture is fully customizable here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Sequence

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# Mapping of activation names (as written in the YAML config) to their nn classes.
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def build_mlp(input_dim: int, hidden_sizes: Sequence[int], activation: str) -> tuple[nn.Sequential, int]:
    """
    Build a multi-layer perceptron trunk with LayerNorm between hidden layers.

    Args:
        input_dim: Number of input features (size of the observation vector).
        hidden_sizes: Width of each hidden layer, in order.
        activation: Activation name, one of the keys of :data:`_ACTIVATIONS`.

    Returns:
        A tuple ``(trunk, output_dim)`` where ``trunk`` is the ``nn.Sequential`` module
        and ``output_dim`` is the width of its last hidden layer (the feature size to
        feed into a downstream head).

    Raises:
        KeyError: If ``activation`` is not a supported activation name.
    """
    activation_cls = _ACTIVATIONS[activation.lower()]
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))  # custom touch: stabilizes training vs. plain MLP
        layers.append(activation_cls())
        last_dim = hidden_dim
    return nn.Sequential(*layers), last_dim


class CustomGaussianPolicy(GaussianMixin, Model):
    """
    Custom Gaussian policy network for PPO.

    The mean of the action distribution is produced by a custom MLP trunk followed by a
    linear head; the log standard deviation is a learnable per-action parameter.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_sizes: Sequence[int] = (256, 128),
        activation: str = "elu",
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
    ) -> None:
        """
        Initialize the custom Gaussian policy.

        Args:
            observation_space: Environment observation space (passed by skrl).
            action_space: Environment action space (passed by skrl).
            device: Torch device the model lives on.
            hidden_sizes: Width of each hidden layer of the shared trunk.
            activation: Activation function name for the trunk.
            clip_actions: Whether to clip sampled actions to the action space.
            clip_log_std: Whether to clip the log standard deviation.
            min_log_std: Lower bound for the clipped log standard deviation.
            max_log_std: Upper bound for the clipped log standard deviation.
            initial_log_std: Initial value of the learnable log-std parameter.
        """
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.trunk, trunk_out_dim = build_mlp(self.num_observations, hidden_sizes, activation)
        self.mean_layer = nn.Linear(trunk_out_dim, self.num_actions)
        # One learnable log-std per action dimension (state-independent, as in standard PPO).
        self.log_std_parameter = nn.Parameter(initial_log_std * torch.ones(self.num_actions))

    def compute(self, inputs: dict, role: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute the action distribution parameters for the given states.

        Args:
            inputs: Dict with key ``"states"`` holding the (preprocessed) observations.
            role: Model role string supplied by skrl (unused, single-head model).

        Returns:
            A tuple ``(mean_actions, log_std, outputs)`` as expected by skrl's GaussianMixin.
        """
        features = self.trunk(inputs["states"])
        return self.mean_layer(features), self.log_std_parameter, {}


class CustomDeterministicValue(DeterministicMixin, Model):
    """
    Custom deterministic value network (critic) for PPO.

    Estimates the scalar state value V(s) from the observation using the same custom MLP
    trunk architecture as the policy, followed by a single-output linear head.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_sizes: Sequence[int] = (256, 128),
        activation: str = "elu",
        clip_actions: bool = False,
    ) -> None:
        """
        Initialize the custom value network.

        Args:
            observation_space: Environment observation space (passed by skrl).
            action_space: Environment action space (passed by skrl).
            device: Torch device the model lives on.
            hidden_sizes: Width of each hidden layer of the trunk.
            activation: Activation function name for the trunk.
            clip_actions: Whether to clip the output (unused for a value head).
        """
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.trunk, trunk_out_dim = build_mlp(self.num_observations, hidden_sizes, activation)
        self.value_layer = nn.Linear(trunk_out_dim, 1)

    def compute(self, inputs: dict, role: str) -> tuple[torch.Tensor, dict]:
        """
        Compute the state value for the given states.

        Args:
            inputs: Dict with key ``"states"`` holding the (preprocessed) observations.
            role: Model role string supplied by skrl (unused).

        Returns:
            A tuple ``(value, outputs)`` as expected by skrl's DeterministicMixin.
        """
        return self.value_layer(self.trunk(inputs["states"])), {}

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom PPO agent for the Lerobot Ts-Pnp task.

This module subclasses skrl's :class:`~skrl.agents.torch.ppo.PPO` to demonstrate a fully
custom loss function and an explicit, hand-written backpropagation step. The override is
contained entirely in :meth:`CustomPPO._update`; the rollout collection, GAE computation,
and bookkeeping reuse the proven skrl machinery.

Customizations versus stock PPO:

* **Custom loss function** - the total loss is assembled explicitly from
  (1) the clipped PPO surrogate policy loss, (2) a selectable value loss (MSE or Huber),
  (3) an optional entropy bonus, and (4) an optional action-magnitude regularization term
  (``behavior_reg_scale``) that discourages saturating actuators.
* **Custom backpropagation** - the optimization step is written out by hand
  (``zero_grad`` -> ``backward`` -> gradient-norm clipping -> ``step``) instead of going
  through skrl's mixed-precision ``GradScaler`` path, so the gradient flow is explicit and
  easy to modify.

Extra configuration keys (read from the agent config, with defaults):

* ``value_loss_type``     - ``"mse"`` (default) or ``"huber"``.
* ``behavior_reg_scale``  - scaling factor for the action-magnitude penalty (default ``0.0``).
"""

from __future__ import annotations

import copy
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveLR

# Default config = stock PPO defaults plus the extra knobs this custom agent understands.
CUSTOM_PPO_DEFAULT_CONFIG = copy.deepcopy(PPO_DEFAULT_CONFIG)
CUSTOM_PPO_DEFAULT_CONFIG["value_loss_type"] = "mse"  # "mse" or "huber"
CUSTOM_PPO_DEFAULT_CONFIG["behavior_reg_scale"] = 0.0  # action-magnitude penalty weight


class CustomPPO(PPO):
    """PPO variant with a custom composite loss and an explicit backpropagation step."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the custom PPO agent.

        Accepts the same arguments as :class:`skrl.agents.torch.ppo.PPO`. The two extra
        configuration keys (``value_loss_type`` and ``behavior_reg_scale``) are read from
        ``self.cfg`` after the base class has merged the configuration dictionaries.
        """
        super().__init__(*args, **kwargs)

        # Read the custom loss knobs (fall back to defaults if absent from the config).
        self._value_loss_type = str(self.cfg.get("value_loss_type", "mse")).lower()
        self._behavior_reg_scale = float(self.cfg.get("behavior_reg_scale", 0.0))

    def _compute_value_loss(self, returns: torch.Tensor, predicted_values: torch.Tensor) -> torch.Tensor:
        """
        Compute the (unscaled) value-function loss using the configured loss type.

        Args:
            returns: Bootstrapped returns (regression targets).
            predicted_values: Values predicted by the critic.

        Returns:
            The scalar value loss before applying ``value_loss_scale``.

        Raises:
            ValueError: If ``value_loss_type`` is neither ``"mse"`` nor ``"huber"``.
        """
        if self._value_loss_type == "mse":
            return F.mse_loss(returns, predicted_values)
        if self._value_loss_type == "huber":
            return F.smooth_l1_loss(returns, predicted_values)
        raise ValueError(f"Unknown value_loss_type '{self._value_loss_type}' (expected 'mse' or 'huber')")

    def _update(self, timestep: int, timesteps: int) -> None:
        """
        Algorithm's main update step with a custom loss and explicit backpropagation.

        This mirrors skrl PPO's update (GAE, mini-batching, KL early-stopping, LR
        scheduling) but assembles the loss and performs the optimizer step by hand.

        Args:
            timestep: Current global timestep (unused directly, kept for API parity).
            timesteps: Total number of timesteps (unused directly, kept for API parity).
        """

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float,
            lambda_coefficient: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute Generalized Advantage Estimation returns and advantages."""
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]
            for i in reversed(range(memory_size)):
                next_value = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i] - values[i] + discount_factor * not_dones[i] * (next_value + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        # --- compute returns and advantages (bootstrap from the last observed states) ---
        with torch.no_grad():
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_reg_loss = 0.0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:

                sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                _, next_log_prob, _ = self.policy.act(
                    {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                )

                # approximate KL divergence (for KL-adaptive LR and early stopping)
                with torch.no_grad():
                    ratio_log = next_log_prob - sampled_log_prob
                    kl_divergence = ((torch.exp(ratio_log) - 1) - ratio_log).mean()
                    kl_divergences.append(kl_divergence)

                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break

                # ============================================================
                # CUSTOM LOSS FUNCTION
                # ============================================================
                # (1) clipped surrogate policy loss (standard PPO objective)
                ratio = torch.exp(next_log_prob - sampled_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(
                    ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                )
                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # (2) optional entropy bonus (encourages exploration)
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                else:
                    entropy_loss = torch.zeros((), device=self.device)

                # (3) value loss (MSE or Huber, selected via config)
                predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")
                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(
                        predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                    )
                value_loss = self._value_loss_scale * self._compute_value_loss(sampled_returns, predicted_values)

                # (4) custom action-magnitude regularization (keeps actuators away from limits)
                if self._behavior_reg_scale:
                    mean_actions = self.policy.distribution(role="policy").mean
                    reg_loss = self._behavior_reg_scale * mean_actions.pow(2).mean()
                else:
                    reg_loss = torch.zeros((), device=self.device)

                total_loss = policy_loss + entropy_loss + value_loss + reg_loss

                # ============================================================
                # CUSTOM BACKPROPAGATION (explicit, no GradScaler)
                # ============================================================
                self.optimizer.zero_grad()
                total_loss.backward()

                # reduce gradients across workers in distributed runs
                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                # gradient-norm clipping
                if self._grad_norm_clip > 0:
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()),
                            self._grad_norm_clip,
                        )

                self.optimizer.step()

                # accumulate losses for logging
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                if self._behavior_reg_scale:
                    cumulative_reg_loss += reg_loss.item()

            # learning-rate scheduling
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # --- logging ---
        num_updates = self._learning_epochs * self._mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / num_updates)
        self.track_data("Loss / Value loss", cumulative_value_loss / num_updates)
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / num_updates)
        if self._behavior_reg_scale:
            self.track_data("Loss / Action regularization loss", cumulative_reg_loss / num_updates)
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

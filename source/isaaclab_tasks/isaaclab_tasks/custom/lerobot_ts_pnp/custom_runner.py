# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom skrl Runner for the Lerobot Ts-Pnp task.

skrl's stock :class:`~skrl.utils.runner.torch.Runner` resolves model/agent classes only by
the built-in names hard-coded in its ``_component`` method, and it builds models through the
YAML model-instantiator (which expects skrl's generated functions, not hand-written
``nn.Module`` classes). This subclass extends the runner so the task can use:

* the custom networks from :mod:`.custom_network` (``CustomGaussian`` / ``CustomDeterministic``), and
* the custom PPO agent from :mod:`.custom_agent` (``CustomPPO``).

It overrides exactly three hooks:

* :meth:`_component`       - resolve the custom class names (delegating to the base class otherwise).
* :meth:`_generate_models` - instantiate hand-written model classes directly.
* :meth:`_generate_agent`  - route the ``CustomPPO`` config through the base PPO build path.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Type, Union

from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.models.torch import Model
from skrl.utils.runner.torch import Runner

from .custom_agent import CUSTOM_PPO_DEFAULT_CONFIG, CustomPPO
from .custom_network import CustomDeterministicValue, CustomGaussianPolicy


class CustomRunner(Runner):
    """skrl Runner that knows how to build this task's custom networks and PPO agent."""

    def _component(self, name: str) -> Type:
        """
        Resolve a component class from its string identifier.

        Adds the task's custom classes on top of skrl's built-ins; unknown names are
        forwarded to the base-class resolver.

        Args:
            name: Component identifier as written in the YAML config (case-insensitive).

        Returns:
            The resolved class (or default-config dict for ``*_default_config`` names).
        """
        key = name.lower()
        if key == "customgaussian":
            return CustomGaussianPolicy
        if key == "customdeterministic":
            return CustomDeterministicValue
        # Route the PPO agent (and its default config) to the custom implementation so the
        # base-class single-agent PPO build path produces a CustomPPO instance.
        if key in ("ppo", "customppo"):
            return CustomPPO
        if key in ("ppo_default_config", "customppo_default_config"):
            return CUSTOM_PPO_DEFAULT_CONFIG
        return super()._component(name)

    def _generate_models(
        self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any]
    ) -> Mapping[str, Mapping[str, Model]]:
        """
        Instantiate the (separate) custom policy and value models directly.

        Unlike skrl's model-instantiator path, the hand-written ``nn.Module`` classes are
        constructed straight from their config keyword arguments. Only the ``separate``
        (non-shared) model layout used by this task is supported.

        Args:
            env: The wrapped (single-agent) environment.
            cfg: Runner configuration dictionary.

        Returns:
            Nested mapping ``{agent_id: {role: model}}`` expected by the base Runner.

        Raises:
            ValueError: If the ``models`` config is missing or requests shared models.
        """
        device = env.device
        observation_space = env.observation_space
        action_space = env.action_space

        models_cfg = copy.deepcopy(cfg.get("models"))
        if not models_cfg:
            raise ValueError("No 'models' are defined in cfg")

        # This task uses separate policy/value networks only.
        if not models_cfg.pop("separate", True):
            raise ValueError("CustomRunner only supports separate (non-shared) models")

        models: dict[str, dict[str, Model]] = {"agent": {}}
        for role, role_cfg in models_cfg.items():
            role_cfg = dict(role_cfg)
            model_class = self._component(role_cfg.pop("class"))
            models["agent"][role] = model_class(
                observation_space=observation_space,
                action_space=action_space,
                device=device,
                **role_cfg,
            )

        # initialize the models' state dictionaries (skrl bookkeeping)
        for role, model in models["agent"].items():
            model.init_state_dict(role)

        return models

    def _generate_agent(
        self,
        env: Union[Wrapper, MultiAgentEnvWrapper],
        cfg: Mapping[str, Any],
        models: Mapping[str, Mapping[str, Model]],
    ):
        """
        Build the agent, normalizing the custom agent name to PPO's build path.

        The base ``_generate_agent`` only assembles ``agent_kwargs`` for a fixed set of
        known agent names; renaming ``CustomPPO`` to ``PPO`` lets that path run, while
        :meth:`_component` ensures the instantiated class is :class:`CustomPPO`.

        Args:
            env: The wrapped environment.
            cfg: Runner configuration dictionary.
            models: The model instances produced by :meth:`_generate_models`.

        Returns:
            The instantiated :class:`CustomPPO` agent.
        """
        cfg = copy.deepcopy(cfg)
        if cfg.get("agent", {}).get("class", "").lower() == "customppo":
            cfg["agent"]["class"] = "PPO"
        return super()._generate_agent(env, cfg, models)

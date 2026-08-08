"""Offline sanity test for the lerobot_ts_pnp custom network / agent / runner.

Runs WITHOUT Isaac Sim. Loads the three custom modules as a synthetic package (so the
relative imports in custom_runner resolve), then:
  1. checks the custom models forward to the right shapes,
  2. checks CustomRunner._component / _generate_models build models from the YAML cfg,
  3. runs a full CustomPPO._update on a fabricated rollout to exercise the custom loss
     and the explicit backpropagation.
"""

import importlib.util
import os
import sys
import types

import gymnasium as gym
import torch

# This test lives inside the task directory; the custom modules are its siblings.
TASK_DIR = os.path.abspath(os.path.dirname(__file__))

# --- load the custom modules as a synthetic package so relative imports work ---
pkg = types.ModuleType("tspnp")
pkg.__path__ = [TASK_DIR]
sys.modules["tspnp"] = pkg


def _load(name):
    spec = importlib.util.spec_from_file_location(f"tspnp.{name}", os.path.join(TASK_DIR, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"tspnp.{name}"] = module
    spec.loader.exec_module(module)
    return module


custom_network = _load("custom_network")
custom_agent = _load("custom_agent")
custom_runner = _load("custom_runner")

device = "cpu"
obs_space = gym.spaces.Box(-float("inf"), float("inf"), shape=(4,))
act_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))

# ---------------------------------------------------------------- 1. models forward
policy = custom_network.CustomGaussianPolicy(obs_space, act_space, device, hidden_sizes=[32, 32], activation="elu")
value = custom_network.CustomDeterministicValue(obs_space, act_space, device, hidden_sizes=[32, 32], activation="elu")
states = torch.randn(7, 4)
mean, log_std, _ = policy.compute({"states": states}, "policy")
val, _ = value.compute({"states": states}, "value")
assert mean.shape == (7, 1), mean.shape
assert val.shape == (7, 1), val.shape
print("[1] custom models forward OK:", mean.shape, val.shape)

# ---------------------------------------------------------------- 2. runner wiring
runner = custom_runner.CustomRunner.__new__(custom_runner.CustomRunner)  # bypass __init__
assert runner._component("CustomGaussian") is custom_network.CustomGaussianPolicy
assert runner._component("CustomDeterministic") is custom_network.CustomDeterministicValue
assert runner._component("ppo") is custom_agent.CustomPPO
assert runner._component("CustomPPO") is custom_agent.CustomPPO
assert runner._component("ppo_default_config") is custom_agent.CUSTOM_PPO_DEFAULT_CONFIG
# RandomMemory should still resolve via the base class
assert runner._component("RandomMemory").__name__ == "RandomMemory"


class _FakeEnv:
    device = "cpu"
    num_envs = 4
    observation_space = obs_space
    action_space = act_space
    state_space = 0


models_cfg = {
    "separate": True,
    "policy": {"class": "CustomGaussian", "hidden_sizes": [32, 32], "activation": "elu", "initial_log_std": 0.0},
    "value": {"class": "CustomDeterministic", "hidden_sizes": [32, 32], "activation": "elu"},
}
models = runner._generate_models(_FakeEnv(), {"models": models_cfg})
assert isinstance(models["agent"]["policy"], custom_network.CustomGaussianPolicy)
assert isinstance(models["agent"]["value"], custom_network.CustomDeterministicValue)
print("[2] CustomRunner._component / _generate_models OK")

# ---------------------------------------------------------------- 3. custom update
from skrl.memories.torch import RandomMemory

rollouts = 8
num_envs = 4
cfg = dict(custom_agent.CUSTOM_PPO_DEFAULT_CONFIG)
cfg.update(
    {
        "rollouts": rollouts,
        "learning_epochs": 2,
        "mini_batches": 2,
        "grad_norm_clip": 1.0,
        "value_loss_type": "huber",
        "behavior_reg_scale": 0.1,  # exercise the custom regularization term
        "entropy_loss_scale": 0.01,  # exercise the entropy term
    }
)

memory = RandomMemory(memory_size=rollouts, num_envs=num_envs, device=device)
agent = custom_agent.CustomPPO(
    models={"policy": policy, "value": value},
    memory=memory,
    observation_space=obs_space,
    action_space=act_space,
    device=device,
    cfg=cfg,
)
agent.init()

# snapshot params to confirm backprop actually updates weights
before = policy.mean_layer.weight.detach().clone()

# rollout is driven under no_grad, exactly like skrl's trainer does
with torch.no_grad():
    for t in range(rollouts):
        s = torch.randn(num_envs, 4)
        a, logp, _ = agent.act(s, t, 100)
        ns = torch.randn(num_envs, 4)
        r = torch.randn(num_envs, 1)
        term = torch.zeros(num_envs, 1, dtype=torch.bool)
        trunc = torch.zeros(num_envs, 1, dtype=torch.bool)
        agent.record_transition(s, a, r, ns, term, trunc, {}, t, 100)

agent.set_mode("train")
agent._update(rollouts, 100)
agent.set_mode("eval")

after = policy.mean_layer.weight.detach().clone()
changed = not torch.allclose(before, after)
assert changed, "policy weights did not change -> backprop/optimizer step failed"
assert torch.isfinite(after).all(), "non-finite weights after update"
print("[3] CustomPPO._update (custom loss + custom backprop) OK; weights updated:", changed)

print("\nALL OFFLINE TESTS PASSED")

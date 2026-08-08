# Lerobot Ts-Pnp — Custom Network + Custom Loss/Backprop

Task id: `Isaac-Lerobot-Ts-Pnp-Direct-v0`

This task is set up so you can train with a **custom neural network**, a **custom loss
function**, and an **explicit (hand-written) backpropagation step**, while still using the
standard skrl/Isaac Lab training flow.

> The environment itself (`lerobot_ts_pnp_env.py`) is currently the cartpole reference
> environment — it is a small, fast, working environment used to exercise and validate the
> custom training stack. Swap in your real pick-and-place dynamics later; the custom network /
> loss / backprop wiring below is independent of the environment.

## What is custom and where

| Concern | File | Class / hook |
|---|---|---|
| Custom network | `custom_network.py` | `CustomGaussianPolicy`, `CustomDeterministicValue` (hand-written `nn.Module`s with a `LayerNorm` MLP trunk) |
| Custom loss function | `custom_agent.py` | `CustomPPO._update` — assembles policy + value (MSE/Huber) + entropy + action-magnitude regularization |
| Custom backpropagation | `custom_agent.py` | `CustomPPO._update` — explicit `zero_grad → backward → grad-norm clip → step` (no `GradScaler`) |
| Wiring into skrl | `custom_runner.py` | `CustomRunner` overrides `_component`, `_generate_models`, `_generate_agent` |
| Entry point | `train.py` | same as the stock skrl trainer but uses `CustomRunner` |
| Config | `agents/skrl_ppo_cfg.yaml` | selects the custom classes and exposes the loss knobs |

skrl's stock `Runner` only knows its built-in model/agent classes (resolved by name in
`Runner._component`) and builds models via the YAML model-instantiator. `CustomRunner`
extends those three hooks so the hand-written classes are used instead. The task-local
`train.py` is a thin copy of `scripts/reinforcement_learning/skrl/train.py` that swaps in
`CustomRunner`.

## Custom loss configuration

In `agents/skrl_ppo_cfg.yaml`, under `agent:`:

- `value_loss_type` — `"mse"` (default) or `"huber"` for the critic loss.
- `behavior_reg_scale` — weight of an action-magnitude penalty `mean(action_mean²)` that
  discourages saturating actuators (`0.0` disables it).
- The usual PPO knobs (`entropy_loss_scale`, `value_loss_scale`, `ratio_clip`,
  `grad_norm_clip`, …) are read by the custom update too.

## Train

```powershell
# Recommended (matches how the other lerobot tasks are run on this machine)
C:/Users/USER/.conda/envs/isaaclab/python.exe source/isaaclab_tasks/isaaclab_tasks/custom/lerobot_ts_pnp/train.py --task=Isaac-Lerobot-Ts-Pnp-Direct-v0 --headless

# or via the isaaclab launcher
isaaclab -p source/isaaclab_tasks/isaaclab_tasks/custom/lerobot_ts_pnp/train.py --task=Isaac-Lerobot-Ts-Pnp-Direct-v0 --headless
```

Useful flags: `--num_envs N`, `--max_iterations I`, `--seed S`, `--video`.

> The stock `scripts/reinforcement_learning/skrl/train.py` would run this task too, but with
> skrl's default `Runner` it would **ignore** the custom network/loss. Use the task-local
> `train.py` above to get the custom stack.

## Validate / quick test

Smoke-test end-to-end (boots Isaac Sim, runs a few updates):

```powershell
C:/Users/USER/.conda/envs/isaaclab/python.exe source/isaaclab_tasks/isaaclab_tasks/custom/lerobot_ts_pnp/train.py --task=Isaac-Lerobot-Ts-Pnp-Direct-v0 --headless --num_envs 64 --max_iterations 4
```

Sim-free unit check of the custom network / agent / runner (fast, no Isaac Sim):

```powershell
C:/Users/USER/.conda/envs/isaaclab/python.exe source/isaaclab_tasks/isaaclab_tasks/custom/lerobot_ts_pnp/offline_test.py
```

The offline test verifies the model forward shapes, that `CustomRunner` resolves/builds the
custom models, and that one `CustomPPO._update` runs the custom loss and actually updates the
policy weights via the explicit backprop step.

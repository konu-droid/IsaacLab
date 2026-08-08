#!/usr/bin/env python3
"""Export an skrl policy trained in Isaac Lab to ONNX.

Two modes:

  1. Isaac Lab mode (default) -- builds the real environment so the observation/
     action spaces and the model architecture are guaranteed to match training:

         ./isaaclab.sh -p export_skrl_onnx.py \
             --task Isaac-Velocity-Flat-Anymal-C-v0 \
             --checkpoint logs/skrl/anymal_c_flat/<run>/checkpoints/best_agent.pt

  2. Offline mode -- no Isaac Sim, no GPU required. Rebuilds the model from the
     agent YAML that Isaac Lab dumped next to the run and infers the observation
     and action sizes from the checkpoint itself:

         python export_skrl_onnx.py --offline \
             --checkpoint logs/skrl/anymal_c_flat/<run>/checkpoints/best_agent.pt

Tasks that define their own networks/agent (their agent YAML names classes such as
`CustomGaussian` rather than skrl's built-ins) additionally need the task's Runner
subclass, which knows how to build them:

        ./isaaclab.sh -p export_skrl_onnx.py \
            --task Isaac-Lerobot-Ts-Pnp-Direct-v0 \
            --runner isaaclab_tasks.custom.lerobot_ts_pnp.custom_runner:CustomRunner \
            --checkpoint logs/skrl/<run>/checkpoints/best_agent.pt

The exported graph maps *raw* observations to *deterministic* actions:

    raw obs -> RunningStandardScaler -> policy network -> mean action

so nothing outside the ONNX file needs to know about normalization statistics.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import shutil
import subprocess
import sys
import tempfile

# Task environments may call `wandb.init()` from their constructor. Exporting a
# policy is not a training run, so keep it from creating (and uploading) a run.
# `setdefault` leaves an explicit user setting alone.
os.environ.setdefault("WANDB_MODE", "disabled")

# ----------------------------------------------------------------------------- CLI

parser = argparse.ArgumentParser(description="Export an skrl policy to ONNX.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the skrl checkpoint (.pt).")
parser.add_argument("--offline", action="store_true", help="Do not launch Isaac Sim; rebuild from YAML.")
parser.add_argument("--task", type=str, default=None, help="Task name (Isaac Lab mode only).")
parser.add_argument("--entry-point", type=str, default="skrl_cfg_entry_point",
                    help="Gym registry key for the skrl config (e.g. skrl_amp_cfg_entry_point).")
parser.add_argument("--runner", type=str, default=None,
                    help="Custom skrl Runner as 'module.path:ClassName', for tasks whose agent YAML "
                         "names hand-written model/agent classes (e.g. CustomGaussian).")
parser.add_argument("--config", type=str, default=None, help="Agent YAML (offline mode). Default: <run>/params/agent.yaml")
parser.add_argument("--num-obs", type=int, default=None, help="Override observation size (offline mode).")
parser.add_argument("--num-actions", type=int, default=None, help="Override action size (offline mode).")
parser.add_argument("--output", type=str, default=None, help="Output .onnx path. Default: <run>/exported/policy.onnx")
parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
parser.add_argument("--clip-actions", type=float, default=None, help="Symmetrically clamp the exported action to +/- this.")
parser.add_argument("--no-verify", action="store_true", help="Skip the onnxruntime parity check.")

if "--offline" not in sys.argv:
    # Isaac Lab needs its launcher args registered and the app started before any
    # isaaclab / omni import happens.
    try:
        from isaaclab.app import AppLauncher  # Isaac Lab 2.x
    except ImportError:
        from omni.isaac.lab.app import AppLauncher  # Isaac Lab 1.x

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
else:
    args = parser.parse_args()
    simulation_app = None

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# ----------------------------------------------------------------------------- wrapper


class OnnxPolicy(nn.Module):
    """raw observation -> deterministic (mean) action, normalization included."""

    def __init__(self, policy, preprocessor=None, clip: float | None = None):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.clip = clip

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.preprocessor is not None:
            obs = self.preprocessor(obs, train=False)
        # Both keys are supplied so this works on skrl 1.x (`inputs["states"]`)
        # and skrl 2.x (`inputs["observations"]`), and whichever the YAML's
        # `input:` field selected.
        inputs = {"states": obs, "observations": obs}
        # compute() returns the *mean* action as its first element on both skrl
        # 1.x (mean, log_std, outputs) and 2.x (mean, {"log_std": ...}). Going
        # through act() instead would trace the Normal distribution and the
        # rsample() into the graph.
        actions = self.policy.compute(inputs, role="policy")[0]
        if self.clip is not None:
            actions = torch.clamp(actions, -self.clip, self.clip)
        return actions


def call_filtered(fn, /, **kwargs):
    """Call `fn` with only the kwargs it actually accepts (skrl 1.x vs 2.x)."""
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in params})


def import_object(spec: str):
    """
    Import an object from a ``'module.path:AttributeName'`` specification.

    Args:
        spec: Dotted module path and attribute name separated by a colon.

    Returns:
        The imported attribute (typically a class).

    Raises:
        SystemExit: If the spec is malformed, the module cannot be imported, or
            the attribute does not exist.
    """
    module_name, sep, attribute = spec.partition(":")
    if not sep or not module_name or not attribute:
        raise SystemExit(f"Expected 'module.path:ClassName', got '{spec}'.")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"Could not import '{module_name}': {exc}") from None
    if not hasattr(module, attribute):
        raise SystemExit(f"'{module_name}' has no attribute '{attribute}'.")
    return getattr(module, attribute)


# ----------------------------------------------------------------------------- mode 1: Isaac Lab


def build_from_isaaclab(args):
    """Create the env exactly like play.py does and let skrl's Runner build the agent."""
    import gymnasium as gym
    from skrl.utils.runner.torch import Runner

    try:  # Isaac Lab 2.x
        import isaaclab_tasks  # noqa: F401  (registers the tasks)
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    except ImportError:  # Isaac Lab 1.x
        import omni.isaac.lab_tasks  # noqa: F401
        from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
        from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

    if not args.task:
        raise SystemExit("--task is required unless you pass --offline")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    agent_cfg = load_cfg_from_registry(args.task, args.entry_point)

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # Tasks with hand-written networks/agents ship their own Runner subclass; using it
    # here is what makes the rebuilt model match the one that produced the checkpoint.
    runner_cls = import_object(args.runner) if args.runner else Runner
    runner = runner_cls(env, agent_cfg)
    runner.agent.load(args.checkpoint)
    runner.agent.set_running_mode("eval")

    # Trace on a real observation rather than zeros: in-distribution and it
    # proves the graph accepts what the env actually produces.
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs = obs.get("policy", next(iter(obs.values())))
    return runner.agent, obs[:1].contiguous(), env


# ----------------------------------------------------------------------------- mode 2: offline


def infer_sizes(ckpt) -> tuple[int, int]:
    """Recover observation / action sizes from the checkpoint tensors."""
    num_obs = num_actions = None

    for key in ("state_preprocessor", "observation_preprocessor"):
        sd = ckpt.get(key)
        if isinstance(sd, dict) and "running_mean" in sd:
            num_obs = int(sd["running_mean"].shape[-1])
            break

    policy_sd = ckpt.get("policy") or ckpt.get("models", {}).get("policy")
    if policy_sd is None:
        raise SystemExit(f"No 'policy' entry in checkpoint. Found keys: {list(ckpt)}")

    if "log_std_parameter" in policy_sd:
        num_actions = int(policy_sd["log_std_parameter"].shape[-1])

    weights = [(k, v) for k, v in policy_sd.items() if k.endswith(".weight") and v.dim() == 2]
    if num_obs is None and weights:
        num_obs = int(weights[0][1].shape[1])
    if num_actions is None and weights:
        num_actions = int(weights[-1][1].shape[0])

    if not num_obs or not num_actions:
        raise SystemExit("Could not infer sizes; pass --num-obs and --num-actions explicitly.")
    return num_obs, num_actions


def resolve_custom_model_class(class_name: str, runner_spec: str | None):
    """
    Resolve a model class that skrl's built-in instantiators do not cover.

    Custom tasks name their own ``nn.Module`` policies in the agent YAML (e.g.
    ``CustomGaussian``). Only that task's Runner subclass knows how to map such a
    name onto a class, so ``--runner`` is what makes offline export possible there.

    Args:
        class_name: The ``class:`` value read from the agent YAML.
        runner_spec: ``'module.path:RunnerClass'`` from ``--runner``, or None.

    Returns:
        The resolved model class.

    Raises:
        SystemExit: If no runner was supplied, or it cannot resolve the name.
    """
    if not runner_spec:
        raise SystemExit(
            f"Unknown model class '{class_name}' in the agent config -- it is not one of skrl's "
            "built-in model types, so it must be defined by the task. Either re-run with "
            "--runner <module.path>:<RunnerClass> (the task's skrl Runner subclass), or drop "
            "--offline and pass --task so the task's own code builds the model."
        )
    runner_cls = import_object(runner_spec)
    # `_component` is a pure name -> class lookup, so bypass __init__ (which would
    # demand a live environment) and call it on an uninitialized instance.
    resolved = runner_cls.__new__(runner_cls)._component(class_name)
    if resolved is None:
        raise SystemExit(f"{runner_spec} could not resolve model class '{class_name}'.")
    return resolved


def build_from_yaml(args):
    """Rebuild policy + preprocessor from the agent YAML, no simulator involved."""
    import gymnasium as gym
    import yaml
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.utils.model_instantiators.torch import (
        categorical_model,
        deterministic_model,
        gaussian_model,
        multivariate_gaussian_model,
        shared_model,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    cfg_path = args.config
    if cfg_path is None:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
        cfg_path = os.path.join(run_dir, "params", "agent.yaml")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"Agent config not found at {cfg_path}. Pass it with --config.")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if args.num_obs and args.num_actions:
        num_obs, num_actions = args.num_obs, args.num_actions
    else:
        inferred_obs, inferred_actions = infer_sizes(ckpt)
        num_obs = args.num_obs or inferred_obs
        num_actions = args.num_actions or inferred_actions
    print(f"[info] num_obs={num_obs}  num_actions={num_actions}  config={cfg_path}")

    inf = float("inf")
    obs_space = gym.spaces.Box(-inf, inf, shape=(num_obs,))
    act_space = gym.spaces.Box(-inf, inf, shape=(num_actions,))

    instantiators = {
        "gaussianmixin": gaussian_model,
        "gaussian": gaussian_model,
        "multivariategaussianmixin": multivariate_gaussian_model,
        "deterministicmixin": deterministic_model,
        "deterministic": deterministic_model,
        "categoricalmixin": categorical_model,
    }

    models_cfg = {k: v for k, v in cfg["models"].items() if k != "separate"}
    separate = cfg["models"].get("separate", True)
    common = dict(observation_space=obs_space, state_space=obs_space,
                  action_space=act_space, device="cpu")

    if separate:
        role_cfg = dict(models_cfg["policy"])
        class_name = role_cfg.pop("class")
        instantiator = instantiators.get(class_name.lower())
        if instantiator is not None:
            policy = call_filtered(instantiator, **common, **role_cfg)
        else:
            # Hand-written model class: constructed directly from its YAML kwargs, the
            # same way the task's own Runner does it.
            model_cls = resolve_custom_model_class(class_name, args.runner)
            policy = model_cls(observation_space=obs_space, action_space=act_space,
                               device="cpu", **role_cfg)
        roles = ["policy"]
    else:
        roles = list(models_cfg)
        structure, parameters = [], []
        for role in roles:
            role_cfg = dict(models_cfg[role])
            structure.append(role_cfg.pop("class"))
            parameters.append(role_cfg)
        policy = call_filtered(shared_model, **common, structure=structure,
                               roles=roles, parameters=parameters)

    # materialize LazyLinear layers -- a shared model needs one pass per role,
    # otherwise the value head stays uninitialized and load_state_dict fails
    for role in roles:
        policy.init_state_dict(role=role)
    policy.load_state_dict(ckpt["policy"] if "policy" in ckpt else ckpt["models"]["policy"])
    policy.eval()

    preprocessor = None
    for key in ("state_preprocessor", "observation_preprocessor"):
        if isinstance(ckpt.get(key), dict) and "running_mean" in ckpt[key]:
            preprocessor = RunningStandardScaler(size=num_obs, device="cpu")
            preprocessor.load_state_dict(ckpt[key])
            preprocessor.eval()
            print(f"[info] loaded '{key}' (RunningStandardScaler) into the graph")
            break
    if preprocessor is None:
        print("[warn] no observation normalizer in the checkpoint - assuming none was used")

    return policy, preprocessor, torch.zeros(1, num_obs)


# ----------------------------------------------------------------------------- main


def main():
    env = None
    if args.offline:
        policy, preprocessor, example = build_from_yaml(args)
    else:
        agent, example, env = build_from_isaaclab(args)
        policy = agent.policy
        preprocessor = getattr(agent, "_state_preprocessor", None)
        if preprocessor is None:
            preprocessor = getattr(agent, "_observation_preprocessor", None)

    device = next(policy.parameters()).device
    example = example.to(device)
    num_obs = example.shape[-1]

    model = OnnxPolicy(policy, preprocessor, args.clip_actions).to(device).eval()

    out_path = args.output
    if out_path is None:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
        out_path = os.path.join(run_dir, "exported", "policy.onnx")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with torch.no_grad():
        reference = model(example)
    print(f"[info] traced shape: {tuple(example.shape)} -> {tuple(reference.shape)}")

    export_kwargs = dict(
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=args.opset,
        export_params=True,
        do_constant_folding=True,
    )
    # torch >= 2.9 defaults to the dynamo exporter, which needs `onnxscript`.
    # The TorchScript path has no extra dependency and is what Isaac Lab's own
    # rsl_rl exporter uses, so prefer it when the argument exists.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(model, example, out_path, **export_kwargs)
    print(f"[ok] wrote {out_path}")

    if not args.no_verify:
        verify(model, out_path, num_obs, device)

    if env is not None:
        env.close()
    if simulation_app is not None:
        simulation_app.close()


# Child-process parity checker. Kept as source text rather than a function because it
# must run in a *fresh* interpreter: loading onnxruntime's native library inside a live
# Isaac Sim process raises a Windows access violation, which the old in-process check
# mistook for "onnxruntime not installed" and silently skipped.
_VERIFIER_SOURCE = '''\
"""Compare an exported ONNX graph against reference outputs captured from PyTorch."""

import sys

import numpy as np

# Exit code 3 means "could not check", as opposed to 1 which means "outputs differ".
try:
    import onnxruntime as ort
except Exception as exc:  # ImportError, or a native loader failure
    print(f"[warn] onnxruntime unavailable ({type(exc).__name__}: {exc}) - skipping verification")
    sys.exit(3)

onnx_path, data_path, tolerance = sys.argv[1], sys.argv[2], float(sys.argv[3])
data = np.load(data_path)
observations, reference = data["observations"], data["reference"]

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
actual = session.run(None, {"obs": observations})[0]

abs_err = float(np.abs(actual - reference).max())
rel_err = abs_err / max(1e-6, float(np.abs(reference).max()))
ok = rel_err < tolerance
print(f"[{'ok' if ok else 'FAIL'}] batch of {observations.shape[0]}: "
      f"max abs err {abs_err:.3e}, max rel err {rel_err:.3e}")
sys.exit(0 if ok else 1)
'''


def verify(model, path, num_obs, device, tolerance: float = 1e-4):
    """
    Check that the exported ONNX graph reproduces the PyTorch policy.

    Reference outputs are computed here (where torch and the model live) and the
    ONNX side is evaluated in a separate interpreter, which is both closer to how
    the file will be deployed and the only way to load onnxruntime safely while
    Isaac Sim is running in this process.

    Args:
        model: The traced `OnnxPolicy` wrapper.
        path: Path to the exported .onnx file.
        num_obs: Observation size, used to shape the random probe batch.
        device: Torch device the model lives on.
        tolerance: Maximum acceptable relative error.

    Raises:
        SystemExit: If the ONNX outputs do not match the PyTorch outputs.
    """
    import numpy as np

    # Fixed seed so a failure can be reproduced exactly.
    observations = np.random.default_rng(0).standard_normal((8, num_obs), dtype=np.float32)
    with torch.no_grad():
        reference = model(torch.from_numpy(observations).to(device)).cpu().numpy()

    work_dir = tempfile.mkdtemp(prefix="skrl_onnx_verify_")
    try:
        data_path = os.path.join(work_dir, "parity.npz")
        script_path = os.path.join(work_dir, "verify_onnx.py")
        np.savez(data_path, observations=observations, reference=reference)
        with open(script_path, "w") as handle:
            handle.write(_VERIFIER_SOURCE)
        # The child writes straight to the shared stdout, so drain our buffer first
        # or its verdict appears before the lines that led up to it.
        sys.stdout.flush()
        completed = subprocess.run(
            [sys.executable, script_path, os.path.abspath(path), data_path, str(tolerance)],
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if completed.returncode == 3:
        return  # the child already explained why it could not check
    if completed.returncode != 0:
        raise SystemExit("ONNX output does not match PyTorch - do not deploy this file.")


if __name__ == "__main__":
    main()

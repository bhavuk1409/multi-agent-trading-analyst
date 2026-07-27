"""
scripts/export_rl_weights.py — Export trained PPO policy weights to numpy
=========================================================================
Run this ONCE after training (scripts/train_rl_agent.py) to generate the
lightweight numpy artefact used by src/rl_agent.py at inference time.

    python scripts/export_rl_weights.py

Output
------
  models/rl_policy_weights.npz   (≈21 KB)

This file is the ONLY model artefact needed at inference time.
The original models/rl_policy_ppo.zip (≈4 MB, requires torch/sb3) is kept
for retraining but is NOT imported by the production API.

Dependency graph
----------------
  Training only:  stable-baselines3, gymnasium, torch  → requirements-training.txt
  Inference only: numpy                                 → requirements.txt (already)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

MODEL_DIR    = ROOT / "models"
SB3_ZIP      = MODEL_DIR / "rl_policy_ppo.zip"
WEIGHTS_PATH = MODEL_DIR / "rl_policy_weights.npz"


def export():
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("ERROR: stable-baselines3 not installed.")
        print("       Run: pip install -r requirements-training.txt")
        sys.exit(1)

    if not SB3_ZIP.exists():
        print(f"ERROR: Model not found at {SB3_ZIP}")
        print("       Run scripts/train_rl_agent.py first.")
        sys.exit(1)

    print(f"Loading PPO policy from {SB3_ZIP} …")
    model = PPO.load(str(SB3_ZIP))
    sd    = model.policy.state_dict()

    def w(key):
        return sd[key].detach().cpu().numpy()

    np.savez(
        str(WEIGHTS_PATH),
        W0 = w("mlp_extractor.policy_net.0.weight"),   # (64, 10)
        b0 = w("mlp_extractor.policy_net.0.bias"),     # (64,)
        W1 = w("mlp_extractor.policy_net.2.weight"),   # (64, 64)
        b1 = w("mlp_extractor.policy_net.2.bias"),     # (64,)
        Wa = w("action_net.weight"),                    # (3, 64)
        ba = w("action_net.bias"),                      # (3,)
    )

    size = WEIGHTS_PATH.stat().st_size
    print(f"✓ Saved {WEIGHTS_PATH}  ({size:,} bytes = {size/1024:.1f} KB)")

    # --- Sanity check: numpy vs torch on 100 random obs -----------------
    import torch

    def softmax(x):
        e = np.exp(x - x.max()); return e / e.sum()

    npz = np.load(str(WEIGHTS_PATH))

    max_delta = 0.0
    for _ in range(100):
        obs = np.random.randn(model.observation_space.shape[0]).astype(np.float32)
        # numpy
        h = np.tanh(npz["W0"] @ obs + npz["b0"])
        h = np.tanh(npz["W1"] @ h   + npz["b1"])
        p_np = softmax(npz["Wa"] @ h + npz["ba"])
        # torch
        with torch.no_grad():
            t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32)
            p_t = model.policy.get_distribution(t).distribution.probs.squeeze(0).numpy()
        max_delta = max(max_delta, abs(p_np - p_t).max())

    print(f"✓ Sanity check (100 random obs): max |numpy−torch| = {max_delta:.2e}")
    if max_delta > 1e-5:
        print("  WARNING: delta exceeds 1e-5 — check network architecture assumptions.")
    else:
        print("  Weights exported correctly.")


if __name__ == "__main__":
    export()

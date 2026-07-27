## Summary

<!-- One-line description of what this PR does. -->

## Type

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor
- [ ] Documentation
- [ ] Tests

## Changes

<!-- List the files changed and what was done in each. -->

-
-

## Testing

<!-- How did you test this? Which `make` targets did you run? -->

- [ ] `make test` — all tests pass
- [ ] `cd frontend && npx tsc --noEmit` — no TypeScript errors
- [ ] Manually tested locally with `make dev`

## RL-specific (if applicable)

- [ ] Observation space unchanged, OR `models/rl_policy_weights.npz` updated
- [ ] `scripts/export_rl_weights.py` validated (max |Δ| < 1e-5)
- [ ] `models/rl_obs_stats.json` committed alongside updated weights

## Checklist

- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No secrets, `.env`, or large binary files (>1 MB) committed
- [ ] `config/config.yaml` agent weights still sum to 1.0

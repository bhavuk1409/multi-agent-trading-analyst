# Contributing to NEXUS

Thank you for considering a contribution! Below are the guidelines to keep
the codebase clean and consistent.

---

## Getting started

```bash
git clone https://github.com/bhavuk1409/multi-agent-trading-analyst.git
cd multi-agent-trading-analyst

cp .env.example .env        # add your GROQ_API_KEY
make install                 # create venv + install all deps
make dev                     # start API server + frontend
```

---

## Branch & commit convention

| Type | Branch prefix | Example |
|---|---|---|
| New feature | `feat/` | `feat/add-options-agent` |
| Bug fix | `fix/` | `fix/coordinator-fallback` |
| Docs | `docs/` | `docs/update-api-reference` |
| Refactor | `refactor/` | `refactor/data-handler-cleanup` |
| Tests | `test/` | `test/rl-env-edge-cases` |

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add options flow agent with Black-Scholes surface
fix: prevent coordinator fallback on clean 'unavailable' mention
docs: update RL training workflow in README
```

---

## Development workflow

```bash
make install        # set up venv + install prod + training deps
make dev            # start both servers (http://localhost:5174)
make test           # run all 45 tests
make lint           # ruff check + tsc --noEmit
make format         # ruff format src/ scripts/ tests/
```

---

## Testing

Every PR must keep the test suite green:

```bash
make test
# or individually:
.venv/bin/python -m pytest tests/test_rl_env.py -v        # offline, no keys
.venv/bin/python -m pytest tests/test_data_handler.py -v  # needs network
.venv/bin/python -m pytest tests/test_multi_agent_system.py -v  # needs GROQ_API_KEY
```

If you add a new agent or change the observation space:
- Update `src/rl_env.py` and re-run `scripts/train_rl_agent.py`
- Re-export weights: `python scripts/export_rl_weights.py`
- Commit the updated `models/rl_policy_weights.npz` and `models/rl_obs_stats.json`

---

## Code style

- **Python:** [Ruff](https://github.com/astral-sh/ruff) for linting and
  formatting. Line length 100. Run `make format` before committing.
- **TypeScript:** ESLint + Prettier (via Vite scaffold). Run `cd frontend && npm run lint`.
- **No LangChain** — the project talks to Groq directly via the `openai` SDK
  to keep the Vercel bundle lean. Please don't add LangChain dependencies.
- **No torch at runtime** — if you extend the RL pipeline, keep inference in
  `src/rl_agent.py` using only numpy. torch/sb3 belong in
  `requirements-training.txt` only.

---

## Pull request checklist

- [ ] Tests pass (`make test`)
- [ ] TypeScript compiles (`cd frontend && npx tsc --noEmit`)
- [ ] If you changed agent weights, `config/config.yaml` weights sum to 1.0
- [ ] If you changed the RL observation space, re-export weights and commit the `.npz`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No secrets or `.env` committed

---

## Reporting issues

Use the GitHub issue templates:
- 🐛 **Bug report** — unexpected behaviour, crashes, wrong output
- 💡 **Feature request** — new agent type, new indicator, new UI panel

---

## License

By contributing, you agree that your contributions will be licensed under
the [MIT License](LICENSE).

# NEXUS — Makefile
# =================
# Unified development interface. Run `make help` to see all targets.

PYTHON     := .venv/bin/python
PIP        := .venv/bin/pip
PYTEST     := .venv/bin/python -m pytest
RUFF       := .venv/bin/ruff

.DEFAULT_GOAL := help

# ── Colours ───────────────────────────────────────────────────────────────────
CYAN  := \033[0;36m
GREEN := \033[0;32m
RESET := \033[0m

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  $(CYAN)NEXUS — Multi-Agent Trading Analyst$(RESET)"
	@echo ""
	@echo "  $(GREEN)Setup$(RESET)"
	@echo "    make install          Create .venv and install all production deps"
	@echo "    make install-training Also install RL training deps (sb3, torch)"
	@echo ""
	@echo "  $(GREEN)Development$(RESET)"
	@echo "    make dev              Start API server + React frontend (http://localhost:5174)"
	@echo "    make server           Start API server only (http://localhost:8000)"
	@echo "    make frontend         Start frontend dev server only"
	@echo ""
	@echo "  $(GREEN)Testing & Quality$(RESET)"
	@echo "    make test             Run all 45 tests"
	@echo "    make test-offline     Run only offline tests (no API keys needed)"
	@echo "    make lint             ruff check + TypeScript tsc --noEmit"
	@echo "    make format           ruff format src/ scripts/ tests/"
	@echo ""
	@echo "  $(GREEN)RL Quant Model$(RESET)"
	@echo "    make rl-train         Train PPO policy (requires training deps)"
	@echo "    make rl-export        Export torch weights → numpy .npz for inference"
	@echo "    make rl-backtest      Run two-window policy vs buy-and-hold comparison"
	@echo ""
	@echo "  $(GREEN)Utilities$(RESET)"
	@echo "    make clean            Remove __pycache__, .pytest_cache, build artefacts"
	@echo "    make help             Show this message"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
.PHONY: install
install:
	@echo "$(CYAN)▶  Creating virtual environment …$(RESET)"
	python3 -m venv .venv
	@echo "$(CYAN)▶  Installing production dependencies …$(RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(CYAN)▶  Installing frontend dependencies …$(RESET)"
	cd frontend && npm install
	@echo "$(GREEN)✓  Done. Copy .env.example → .env and add your GROQ_API_KEY.$(RESET)"

.PHONY: install-training
install-training: install
	@echo "$(CYAN)▶  Installing RL training dependencies …$(RESET)"
	$(PIP) install -r requirements-training.txt
	@echo "$(GREEN)✓  Training deps installed (stable-baselines3, torch, gymnasium).$(RESET)"

# ── Development ───────────────────────────────────────────────────────────────
.PHONY: dev
dev:
	@./start.sh

.PHONY: server
server:
	$(PYTHON) api_server.py

.PHONY: frontend
frontend:
	cd frontend && npm run dev

# ── Testing & Quality ─────────────────────────────────────────────────────────
.PHONY: test
test:
	$(PYTEST) tests/ -v

.PHONY: test-offline
test-offline:
	$(PYTEST) tests/test_rl_env.py -v

.PHONY: lint
lint:
	$(RUFF) check src/ scripts/ tests/ api/ api_server.py
	cd frontend && npx tsc --noEmit
	@echo "$(GREEN)✓  Lint passed.$(RESET)"

.PHONY: format
format:
	$(RUFF) format src/ scripts/ tests/ api/ api_server.py
	@echo "$(GREEN)✓  Formatted.$(RESET)"

# ── RL Quant Model ────────────────────────────────────────────────────────────
.PHONY: rl-train
rl-train:
	@echo "$(CYAN)▶  Training PPO policy (this may take several minutes) …$(RESET)"
	$(PYTHON) scripts/train_rl_agent.py
	@echo "$(GREEN)✓  Training complete. Run 'make rl-export' to update inference weights.$(RESET)"

.PHONY: rl-export
rl-export:
	@echo "$(CYAN)▶  Exporting weights → models/rl_policy_weights.npz …$(RESET)"
	$(PYTHON) scripts/export_rl_weights.py
	@echo "$(GREEN)✓  Export complete. Commit models/rl_policy_weights.npz and models/rl_obs_stats.json.$(RESET)"

.PHONY: rl-backtest
rl-backtest:
	$(PYTHON) scripts/window_backtest.py

# ── Utilities ─────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" | xargs rm -rf
	find . -type d -name .pytest_cache -not -path "./.venv/*" | xargs rm -rf
	find . -type f -name "*.pyc" -not -path "./.venv/*" | xargs rm -f
	rm -rf frontend/dist frontend/.vite
	@echo "$(GREEN)✓  Cleaned.$(RESET)"

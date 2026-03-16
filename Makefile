# ============================================================
# MALTO Text Authorship Detection — Makefile
# ============================================================
# Usage:
#   make train    — Run full training pipeline
#   make infer    — Run inference on test set
#   make submit   — Generate Kaggle submission
#   make test     — Run all unit tests
#   make test-cov — Run tests with coverage report
#   make clean    — Remove Python cache files

CONFIG ?= configs/config.yaml

.PHONY: train infer submit test test-cov clean help

train:
	@echo "Starting training pipeline..."
	python main_train.py --config $(CONFIG)

infer:
	@echo "Running inference..."
	python main_infer.py --config $(CONFIG)

submit:
	@echo "Generating submission..."
	python main_submit.py --config $(CONFIG)

test:
	@echo "Running tests..."
	python -m pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	@echo "Cleaning Python cache..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true

help:
	@echo ""
	@echo "MALTO Text Authorship Detection"
	@echo "================================"
	@echo "  make train    - Full training pipeline (CV + final model)"
	@echo "  make infer    - Inference on test set"
	@echo "  make submit   - Generate Kaggle submission CSV"
	@echo "  make test     - Run unit tests"
	@echo "  make test-cov - Run tests with coverage"
	@echo "  make clean    - Remove __pycache__ and .pyc files"
	@echo ""
	@echo "  Override config: make train CONFIG=configs/my_config.yaml"
	@echo ""

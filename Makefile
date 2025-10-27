# Makefile for SusTech Recycling Agent Project

# Environment
PYTHON := poetry run python

.PHONY: help install embed api ui test dev venv

help:
	@echo "SusTech Recycling Agent Makefile Commands:"
	@echo "  install  - Install dependencies"
	@echo "  embed    - Preprocess and embed PDFs into vector DB"
	@echo "  api      - Run FastAPI server locally"
	@echo "  ui       - Run Streamlit frontend locally"
	@echo "  test     - Run similarity evaluator and tests"
	@echo "  eval     - Run design-centered evaluation"
	@echo "  dev      - Instructions for running API and UI concurrently"
	@echo "  venv     - Instructions to activate virtual environment"

install:
	poetry install

embed:
	$(PYTHON) src/Knowledge.py

api:
	PYTHONPATH=. poetry run uvicorn src.serve:app --reload

ui:
	PYTHONPATH=. poetry run streamlit run src/streamlit_ui.py

test:
	@echo "Running local test script..."
	@chmod +x ./src/test_server.sh && ./src/test_server.sh

tests:
	@chmod +x ./src/test_examples.sh && ./src/test_examples.sh && $(PYTHON) src/similarity_evaluator.py

eval:
	$(PYTHON) src/design_evaluator.py

venv:
	@echo "Activate virtual environment with: source .venv/bin/activate"

dev:
	@echo "Run 'make api' and 'make ui' in separate terminals."

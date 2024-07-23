# Makefile

# Define variables
SRC_DIR := src
VENV_DIR := .venv
REQUIREMENTS_FILE := requirements.txt
PYTHON_VERSION := 3.12.4  # Set your preferred Python version

# Default goal
.DEFAULT_GOAL := show_help

# Install Python using asdf
.PHONY: install_python
install_python: ## Install Python using asdf
	@echo "Installing Python $(PYTHON_VERSION) using asdf..."
	asdf install python $(PYTHON_VERSION)
	asdf local python $(PYTHON_VERSION)
	@echo "Python $(PYTHON_VERSION) installed and set as local version"

# Create virtual environment
.PHONY: create_venv
create_venv: ## Create a virtual environment
	@echo "Creating virtual environment in $(VENV_DIR)..."
	python3 -m venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)"

# Activate virtual environment
.PHONY: activate_venv
activate_venv: ## Activate the virtual environment
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV_DIR)/bin/activate"

# Install dependencies
.PHONY: install_deps
install_deps: ## Install dependencies using requirements file
	@echo "Installing dependencies from $(REQUIREMENTS_FILE)..."
	$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS_FILE)
	@echo "Dependencies installed"

# Generate requirements.txt
.PHONY: freeze_deps
freeze_deps: ## Freeze current dependencies to requirements.txt
	@echo "Freezing current dependencies to $(REQUIREMENTS_FILE)..."
	$(VENV_DIR)/bin/pip freeze > $(REQUIREMENTS_FILE)
	@echo "Dependencies frozen in $(REQUIREMENTS_FILE)"

# Clean virtual environment
.PHONY: clean_venv
clean_venv: ## Remove the virtual environment
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed"

# Run CLI
.PHONY: run
run: ## Run the CLI with a command
	@echo "Running CLI with command: $(filter-out $@,$(MAKECMDGOALS)) $(ARGS)"
	PYTHONPATH=$(SRC_DIR) python src/cli/main.py $(filter-out $@,$(MAKECMDGOALS)) $(ARGS)

# Run experiments
.PHONY: run_experiments
run_experiments: ## Run the experiments
	PYTHONPATH=$(SRC_DIR) python src/cli/main.py run-experiments --config src/experiments/experiments.yaml

# Prepare evaluation prompts
.PHONY: prepare_evaluation_prompts
prepare_evaluation_prompts: ## Prepare evaluation prompts
	PYTHONPATH=$(SRC_DIR) python src/cli/main.py prepare-evaluation-prompts --data_dir data/experiments

# Run evaluators
.PHONY: run_evaluators
run_evaluators: prepare_evaluation_prompts ## Run the evaluators
	PYTHONPATH=$(SRC_DIR) python src/cli/main.py run-evaluators --evaluation_dir data/evaluations

# Prepare groundedness prompts
.PHONY: prepare_groundedness_prompts
prepare_groundedness_prompts: ## Prepare groundedness prompts
	PYTHONPATH=$(SRC_DIR) python src/cli/main.py prepare-groundedness-prompts --data_dir data/experiments

# Run groundedness evaluators
.PHONY: run_groundedness_evaluators
run_groundedness_evaluators: prepare_groundedness_prompts ## Run groundedness evaluators
	PYTHONPATH=$(SRC_DIR) python src/cli/main.py run-groundedness-evaluators --evaluation_dir data/evaluations

# Parse evaluations
.PHONY: parse_evaluations
parse_evaluations: ## Parse evaluations
	PYTHONPATH=$(SRC_DIR):$(SRC_DIR)/experiments python $(SRC_DIR)/app/llm/parse_evaluations.py

# Compute evaluations
.PHONY: compute_evaluations
compute_evaluations: parse_evaluations ## Compute evaluations
	PYTHONPATH=$(SRC_DIR):$(SRC_DIR)/experiments python $(SRC_DIR)/app/llm/compute_evaluations.py

# gcloud authentication
.PHONY: gcloud_auth
gcloud_auth: ## gcloud ADC authentication
	@echo "To authenticate on gcloud, run:"
	@echo "gcloud auth application-default login"

# Run full pipeline
.PHONY: run_full_pipeline
run_full_pipeline: run_experiments prepare_evaluation_prompts run_evaluators prepare_groundedness_prompts run_groundedness_evaluators compute_evaluations ## Run the entire pipeline from experiments to evaluations

# Show Makefile help
.PHONY: show_help
show_help: ## Show Makefile commands help
	@echo "Available Makefile commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

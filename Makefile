#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = nem-negative-prices
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) negprice/dataset.py
	$(PYTHON_INTERPRETER) negprice/predispatch.py
	$(PYTHON_INTERPRETER) negprice/rooftop_pv.py
	$(PYTHON_INTERPRETER) negprice/weather.py
	$(PYTHON_INTERPRETER) negprice/processing.py
	$(PYTHON_INTERPRETER) negprice/features.py


## Run the rolling-origin backtest: make backtest NOTE="why this run" [FORCE=1 refits every model]
.PHONY: backtest
backtest:
	NOTE="$(NOTE)" $(PYTHON_INTERPRETER) negprice/backtest.py $(if $(FORCE),--force,)

## Zero-shot Chronos-2 forecasts (approx. 40 min on CPU)
.PHONY: chronos
chronos:
	$(PYTHON_INTERPRETER) -m negprice.chronos_fm

## Battery arbitrage value from the backtest predictions
.PHONY: battery
battery:
	$(PYTHON_INTERPRETER) -m negprice.battery

## Export the tables behind the Power BI dashboard
.PHONY: dashboard
dashboard:
	$(PYTHON_INTERPRETER) -m negprice.dashboard

## Rebuild everything: data, Chronos, backtest, battery, dashboard (approx. 3 h)
.PHONY: all
all: NOTE = full rebuild (make all)
all: data chronos backtest battery dashboard

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

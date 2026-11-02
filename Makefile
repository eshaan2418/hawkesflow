# hawkesflow: common entry points (uv-managed virtualenv in .venv).
PY ?= .venv/bin/python
DATA ?= $(HOME)/quantfactory/_data/hawkesflow
.PHONY: build test bench lint fuzz demo data research clean

build:
	uv venv -q .venv && uv pip install -q -e '.[dev]'

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check hawkesflow tests research

# Randomised property checks (likelihood vs brute force, gradient vs finite differences, recovery).
fuzz:
	$(PY) -m pytest -q -k "brute or gradient or recovery or residuals"

# Aggressor-signed executions from the NASDAQ ITCH sample via tickstore's CSV export.
data:
	bash scripts/extract_trades.sh

research:
	$(PY) research/fit_itch.py $(DATA) research/results/itch_fits.json
	$(PY) research/coverage.py research/results/itch_fits.json research/results/coverage.json 40
	$(PY) research/plot_qq.py $(DATA) research/plots/aapl_qq.png

bench:
	$(PY) research/bench.py bench/results/bench.json

demo:
	$(PY) research/fit_itch.py $(DATA) /dev/stdout AAPL

clean:
	rm -rf .venv .pytest_cache

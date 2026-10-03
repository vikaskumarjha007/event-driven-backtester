PY ?= python3
export PYTHONPATH := src:.

.PHONY: setup test demo

setup:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

demo:
	$(PY) -m edbt.demo

PYTHON ?= python

.PHONY: test ingest eval api

test:
	$(PYTHON) -m pytest

ingest:
	$(PYTHON) scripts/ingest.py

eval:
	$(PYTHON) scripts/evaluate.py

api:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

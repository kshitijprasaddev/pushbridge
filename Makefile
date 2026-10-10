PYTHON ?= python3
PNPM ?= pnpm

.PHONY: help install serve test audit dashboard precision package web-install web-data web-dev web-build web-preview
help:
	@echo "install   Install the package and experiment dependencies"
	@echo "serve     Open the saved report on http://127.0.0.1:8767/"
	@echo "test      Run numerical and physical regression checks"
	@echo "audit     Verify raw hashes, strict holds and dashboard evidence"
	@echo "dashboard Rebuild UI data from the included reference results"
	@echo "precision Re-run the frozen precision comparison in a fresh folder"
	@echo "package   Build a ZIP from the files selected for the GitHub repository"
	@echo "web-install Install the Node frontend dependencies from the lockfile"
	@echo "web-data    Export saved experiment evidence for the Three.js app"
	@echo "web-dev     Start the frontend development server"
	@echo "web-build   Build the static frontend for deployment"
	@echo "web-preview Preview the production frontend build"

install:
	$(PYTHON) -m pip install -e '.[dev,analysis,rl]'
serve:
	$(PYTHON) -m http.server 8767 --bind 127.0.0.1
test:
	$(PYTHON) -m pytest -q
audit:
	$(PYTHON) scripts/audit_submission.py
dashboard:
	$(PYTHON) scripts/build_dashboard.py
precision:
	$(PYTHON) scripts/precision_experiment.py --out results/precision_rerun
package:
	$(PYTHON) scripts/package_submission.py
web-install:
	$(PNPM) --dir web install --frozen-lockfile
web-data:
	$(PYTHON) scripts/export_web_data.py
web-dev:
	$(PNPM) --dir web dev --host 127.0.0.1
web-build:
	$(PNPM) --dir web build
web-preview:
	$(PNPM) --dir web preview --host 127.0.0.1

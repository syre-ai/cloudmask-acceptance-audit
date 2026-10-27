# One command for the whole audit: `make all` (about 1.5 h on an RTX 2070 Super, ~4 GB of data).
# Each step is idempotent and caches under data/; `make clean-results` forces a re-score.

PY := uv run python

.PHONY: all env test data screen predict evaluate sweep subgroups sensitivity perturb mitigate figures docs clean-results

all: env test data screen predict evaluate sweep subgroups sensitivity perturb mitigate figures docs

env:
	uv sync --frozen

test:
	uv run pytest -q

data:            ## CloudSEN12+ test/validation samples (range requests), OCM v4 weights (sha256-checked)
	$(PY) scripts/download_data.py

screen:          ## check OCM's published training metadata against the evaluation images
	$(PY) scripts/screen_ocm_training.py

predict:         ## OCM v4 on L1C and L2A
	$(PY) scripts/run_ocm.py

evaluate:        ## A1: headline metrics, paired differences, scene policy
	$(PY) scripts/evaluate.py

sweep:           ## A2: leak/discard curves, operating points chosen on validation
	$(PY) scripts/sweep.py

subgroups:       ## A3: land cover, cloud regime, continent, sun elevation, season
	$(PY) scripts/subgroups.py

sensitivity:     ## Kappaset MGRS-tile exclusion check
	$(PY) scripts/sensitivity.py

perturb:         ## A4: simulated degradations (GPU)
	$(PY) scripts/perturbations.py

mitigate:        ## A5: mask buffer vs threshold, scene gate
	$(PY) scripts/mitigations.py

figures:
	$(PY) scripts/figures.py
	$(PY) scripts/examples.py

docs:            ## README results block, site write-up, client PDF
	$(PY) scripts/provenance.py
	$(PY) scripts/build_docs.py

clean-results:
	rm -rf data/results results/*.csv

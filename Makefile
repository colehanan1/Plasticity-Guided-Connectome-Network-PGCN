.PHONY: conda cache metrics priors baseline app test format

conda:
conda env create -f environment.yml

cache:
pgcn-cache --out data/cache

metrics:
pgcn-metrics --cache-dir data/cache

priors:
@echo "Structural priors pipeline not yet implemented"

baseline:
@echo "Baseline reservoir training not yet implemented"

app:
@echo "Streamlit app launcher not yet implemented"

test:
pytest -q

format:
ruff check .
black .

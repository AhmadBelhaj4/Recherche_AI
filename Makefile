export PYTHONPATH := .

.PHONY: toy test test-real verify-data lint train

## Generate toy data (200 fake VHH sequences, splits, manifest)
toy:
	python scripts/make_toy_data.py

## Run fast tests (skip real-data tests)
test:
	pytest -q -m "not real"

## Run all tests including real-data tests (requires VHH_DATA set)
test-real:
	pytest -q

## Verify data integrity against manifest
verify-data:
	python scripts/verify_data.py --data-dir toy

## Lint with ruff
lint:
	ruff check .

## Train model (stub — will fail until src/train.py is implemented)
train:
	python src/train.py --config configs/vhh_small.yaml

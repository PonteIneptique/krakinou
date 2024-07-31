# KrakI(n)oU, a secondary metric provider for segmentation results with Kraken

Krakinou provides a nice and easy way to compute metrics with a given model.

## Install

```sh
pip install -r requirements.txt
```

## Run with example data

- `--verbose` will show you the table
- `--output` will save all metrics in a JSON for further exploration

```sh
python krakinou.py ./tests/*.xml --model ./tests/model_99.mlmodel --verbose --output results.json
```
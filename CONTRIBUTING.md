# Contributing

Contributions should preserve the distinction between state intervention,
state readout, source attribution, and behavioral output.

Before opening a pull request:

```bash
python -m compileall -q demo src scripts tests
python scripts/verify_direct44_bundle.py
python scripts/verify_direct44_31b_bundle.py
pytest -q
```

Changes to writer composition, reader calibration, layer selection, or reported
metrics must include a deterministic validation record and disclose whether the
evaluation data were used for model or layer selection. Do not commit model
checkpoints, raw conversations, private datasets, activation dumps, credentials,
or machine-specific paths.

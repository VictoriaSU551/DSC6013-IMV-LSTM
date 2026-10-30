# Contributing

Use a separate branch and a fresh directory for numerical experiments. Never edit completed runs or their source/data bindings to make a verifier accept them. Retain registered seeds, negative findings, units and protocol differences.

Install Python3.12, CPU PyTorch2.6 and `requirements.txt`. Set `PYTHONUTF8=1` and `PYTHONPATH` to the absolute `code` directory. Run `python -B -m pytest tests -q -p no:cacheprovider`. Linux uses native flock; Windows loads the adapter through `code/sitecustomize.py`.

Publish code, configuration, reports and compact evidence. Keep data, third-party source, weights, environments, original papers, credentials and raw logs untracked. `sources.lock.json` supports verified acquisition. `code/verify_final.py --public` checks only the published subset.

See `docs/RUNNING.md` for training/verification and `docs/FINAL_AUDIT.md` for scientific limits. New models, data snapshots or larger searches are new experiments, not corrections of inconvenient results.

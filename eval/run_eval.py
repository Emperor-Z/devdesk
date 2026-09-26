"""Thin entrypoint so the harness runs as `python eval/run_eval.py`.

All logic lives in `devdesk.evaluation` so it's importable and unit-tested.
"""

from devdesk.evaluation.harness import main

if __name__ == "__main__":
    raise SystemExit(main())

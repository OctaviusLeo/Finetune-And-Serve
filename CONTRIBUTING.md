# Contributing

## Checks to run before a commit

The six steps GitHub Actions runs on every push and pull request (`.github/workflows/ci.yml`), on a
GitHub-hosted `ubuntu-latest` runner with no GPU and no secrets. Run them from the repository root
on Linux, macOS or WSL2; in WSL2 prefix each command with `bash scripts/wsl_run.sh`, which selects
the project environment (see "Commands" in the README):

```bash
uvx ruff@0.17.0 check .                               # the ruff version CI pins; rules in pyproject.toml
uv sync --frozen                                      # the locked environment, PyTorch included (several GB)
bash scripts/test_unit.sh                             # unit and property tests, 80% statement coverage per non-trivial module
uv run --frozen python -m fts.smoke_eval              # CPU smoke evaluation on a fixed fixture
uv run --frozen python scripts/build_docs.py --check  # README and Model card match their templates and results
uv run --frozen python scripts/check_traceability.py  # every number in them traces to a generated file
```

Tests that need the GPU or Docker carry the `gpu` and `docker` markers and do not run in CI.

## The README and Model card are generated

Do not edit `README.md` or `MODEL_CARD.md` directly. Edit `docs/README.template.md` or
`docs/MODEL_CARD.template.md`, regenerate with `uv run --frozen python scripts/build_docs.py`, and
commit the template together with the regenerated document; the `--check` step above fails CI when
they differ. Every number in both documents must appear in a committed result file
(`results/*.json`, `results/summary.md`, `results/doc_facts.json`) or in `docs/ENVIRONMENT.md`, so a
new number needs a generated source, not a typed value.

## Results are generated

Never edit a file under `results/` by hand. Each result file records the command, seed and commit
that produced it, and `results/summary.md` and the plots are written by `python scripts/build_summary.py`.
A changed result comes from re-running the matching step in the README's Commands table on a clean
commit. Raw run data and logs stay under `runs/`, which is not committed.

## Dependencies

`pyproject.toml` and `uv.lock` pin every Python package exactly, because those pins are part of the
environment behind every reported number; they change only with a deliberate re-run. Dependabot is
therefore configured for GitHub Actions only (`.github/dependabot.yml`, one grouped pull request a
month). After merging an Actions bump, update the row that names that action's version in
`docs/ENVIRONMENT.md` or `THIRD_PARTY.md`, where one exists, in the same change. A new package needs
its license checked and a row in `THIRD_PARTY.md`.

## Branches

`main` is the default branch and its CI stays green. Dependabot bumps are merged with a merge
commit once their checks pass.

# AGENTS.md

These rules apply to every coding session. Project background is in `README.md`. **Before starting a task, read `CURRENT_STATUS.md`**, and follow the `AGENTS.md` inside `data/` or `preprocessing/` when you work there.

## Architecture: the layers never mix
`data/ → preprocessing/ → model/ → validation/`
- **data/** collects raw data only. **preprocessing/** makes every transformation. **model/** only fits models: no train/test splits and no validation. **validation/** takes any model plus validation data and returns metrics, and it owns splits, metrics and QA/QC.
- Each layer reads only from the layer before it through importable code (e.g. `data/reader.py`). Shared code for a layer goes in its `base.py`.
- Notebooks in `eda/` and `experiments/` import layer code and never redefine it. EDA doesn't belong in any layer.

## Coding rules
1. Use the existing `base.py` interfaces. Don't add new abstractions (base classes, wrappers, config systems) unless the issue asks for them.
2. Search the repo before writing anything. Reuse what's there, and never duplicate a reader, preprocessing step or metric.
3. Stay inside your layer. Never change preprocessing to improve a model's results. If a change has to cross layers, flag it in the PR.
4. Add or update `pytest` tests in `tests/` and run the **full suite** with every change.
5. Do only what the issue asks, as simply as possible.

## Environment
- Use **uv** only. Run things with `uv sync` and `uv run …`. Change dependencies only with `uv add` / `uv remove`, never with pip, and never edit `uv.lock` by hand.
- The team is on both macOS and Windows. Use `pathlib` and no absolute paths.
- **Never commit** data, database files (`*.duckdb`) or secrets. Tokens go in `.env`.

## GitHub
- One issue per task, grouped into epics. Take one issue at a time and reference it in your branch and PR.
- PRs must be self-contained and **under about 500 lines**.
- Issues you create must be short and labeled **`claude`**.
- Before opening a PR, check: did it change a layer it shouldn't have? Did it recreate something that already exists? Did it add an unnecessary abstraction? Do all tests pass?

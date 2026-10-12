# preprocessing/: transformation layer

**Job:** turn raw data into model-ready inputs. Every transformation happens here, and only here.

## Rules
- Read raw data **only through `data/reader.py`**. Never read files, call APIs or query the database directly.
- Write each step as a named function or class that does one thing, with no hidden side effects, so the sequence of steps is clear and visible. Shared step logic goes in `base.py`.
- Put every analytic choice in config instead of hard-coding it. That includes the date cutoff (currently 2017 onward), severity filters, the crash-to-camera matching rule (buffer radius or nearest LION node) and the before/after windows.
- Handle NAs explicitly, and log how many rows each step drops or changes.
- Don't build or validate models, ingest raw data, or run EDA here.

## Inferred camera attributes
No public camera dataset exists, so this layer derives activation dates (from first-issuance changepoints in violation data) and approach or orientation (from ticket patterns). Every inferred field must carry a **`method`** column and a **`confidence`** column, because DOT will audit them.
